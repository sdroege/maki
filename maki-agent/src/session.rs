//! The one vocabulary every driver uses to talk about a persisted session.

use std::collections::HashMap;
use std::sync::Arc;

use arc_swap::ArcSwap;
use maki_providers::{ContentBlock, ContextGauge, Message, Role, TokenUsage};
use maki_storage::StateDir;
use maki_storage::frame::StoredFrame;
use maki_storage::id::SessionRef;
use maki_storage::sessions::{SAVE_FAILED, Session, SessionArchive, SessionClaim, SessionError};
use serde_json::{Value, json};
use tracing::warn;

use crate::agent::{
    History, HistorySnapshot, RunContextBuilder, SharedMessages, publish_live_history,
};
use crate::types::EventSender;
use crate::{AgentRunParams, ToolOutput};

/// The one spelling of a persisted maki session. It cannot live in
/// maki-storage: [`ToolOutput`] is ours, and maki-storage must not depend on us.
pub type StoredSession = Session<Message, TokenUsage, ToolOutput>;

/// The one spelling of an archived pre-compaction window of the same.
pub type StoredArchive = SessionArchive<Message, TokenUsage, ToolOutput>;

/// A transcript a driver starts from. The id comes from the caller, because
/// every entry point resolves one and reports it before the run starts, and a
/// second answer minted here would not be the one the user was told.
pub struct Resumed {
    pub id: SessionRef,
    /// The transcript to run on. It is moved out of [`Self::session`] when
    /// there is one, so the two never hold two copies of it.
    pub history: Vec<Message>,
    /// The provider's last prompt count for `history`, so a resumed run budgets
    /// from a measurement instead of an estimate.
    pub context_size: u32,
    /// The prefix `history` was sent under, so the next request can extend it
    /// instead of starting the cache and the model's reasoning over.
    pub frame: Option<Arc<StoredFrame>>,
    /// The stored session `history` came out of, if the driver read one. The
    /// run writes back into it, so the title, spending and meta survive without
    /// a second parse of the file. `None` when nothing is stored under the id
    /// yet, or when the driver owns the session and persists it itself.
    pub session: Option<StoredSession>,
}

impl Resumed {
    /// A session nothing has been stored for yet. Not [`Default`], because
    /// minting an id is a decision, and reaching for it by accident writes a
    /// transcript under an id nobody reported.
    pub fn fresh() -> Self {
        Self::empty(SessionRef::generate())
    }

    /// A run under `id` with nothing read from disk for it.
    pub fn empty(id: SessionRef) -> Self {
        Self {
            id,
            history: Vec::new(),
            context_size: 0,
            frame: None,
            session: None,
        }
    }

    /// A run continuing a session the driver already loaded.
    pub fn stored(id: SessionRef, mut session: StoredSession) -> Self {
        Self {
            id,
            history: session.drain_messages(),
            context_size: session.meta.context_size,
            frame: session.frame().cloned(),
            session: Some(session),
        }
    }
}

/// Everything a session run owns: what it said, how big that is, and where it
/// is written. [`AgentRunParams`] already wants `history` and `gauge` under one
/// owner, and the store joins them so a driver cannot hold a transcript it
/// never persists.
///
/// The transcript is only reachable through [`Self::turn`], whose guard writes
/// it back when it drops, so "built an agent and forgot to persist it" is not a
/// mistake a driver can make.
pub struct SessionTrack {
    history: History,
    gauge: ContextGauge,
    store: SessionStore,
}

impl SessionTrack {
    /// Built once the provider resolves, so a run that never started leaves no
    /// file behind. The state dir comes from the caller rather than being
    /// resolved here, so the run writes where its history was read from.
    ///
    /// Takes the claim by value, so nobody else can write the session for as
    /// long as this track can.
    ///
    /// Publishes the transcript under the session's id, so
    /// `maki.session.messages` works under a headless driver too.
    pub fn open(resumed: Resumed, claim: SessionClaim, storage: StateDir, cwd: &str) -> Self {
        let mirror: SharedMessages = Arc::new(ArcSwap::from_pointee(HistorySnapshot::default()));
        publish_live_history(resumed.id.id(), &mirror);
        Self {
            store: SessionStore::open(storage, claim, resumed.session, cwd),
            history: History::restored(resumed.history)
                .with_frame(resumed.frame)
                .with_mirror(mirror),
            gauge: ContextGauge::restored(resumed.context_size),
        }
    }

    /// One turn against this transcript. The returned guard is the only way to
    /// reach [`AgentRunParams`] and it persists on drop. An
    /// [`Agent`](crate::Agent) built from it borrows the guard, so borrowck
    /// puts the write after the run without the driver arranging it.
    ///
    /// The spec lands here rather than at the end because a driver picks its
    /// model before it builds the agent. Since this is the only path to a
    /// write, no stored session can carry a spec no turn ran on.
    pub fn turn(&mut self, model_spec: String) -> SessionTurn<'_> {
        self.store.session.set_model(model_spec);
        SessionTurn(self)
    }
}

/// A turn in progress. See [`SessionTrack::turn`].
pub struct SessionTurn<'a>(&'a mut SessionTrack);

impl SessionTurn<'_> {
    pub fn run_params(
        &mut self,
        event_tx: EventSender,
        context: RunContextBuilder,
    ) -> AgentRunParams<'_> {
        AgentRunParams {
            history: &mut self.0.history,
            gauge: &mut self.0.gauge,
            event_tx,
            context,
        }
    }
}

impl Drop for SessionTurn<'_> {
    /// A turn that changed nothing writes nothing. A rewrite would move
    /// `updated_at`, which reorders `--continue`, stamp a model no turn ran
    /// on, and commit the restore-time repair over the only copy of the
    /// transcript.
    fn drop(&mut self) {
        let SessionTrack {
            history,
            gauge,
            store,
        } = &mut *self.0;
        if !history.has_unsaved() {
            return;
        }
        match store.record_turn(history.as_slice(), history.frame(), gauge.size()) {
            Ok(()) => history.mark_saved(),
            // Only headless runs save through here (the TUI has its own
            // writer), so printing cannot tear a drawn frame.
            Err(e) => {
                warn!(error = %e, session_id = %store.session.id, "failed to persist session");
                eprintln!("{SAVE_FAILED}{}: {e}", store.session.id);
            }
        }
    }
}

struct SessionStore {
    dir: StateDir,
    claim: SessionClaim,
    session: StoredSession,
}

impl SessionStore {
    /// Opening touches no file: the driver either read the session already or
    /// there is nothing under the id to read. A session nothing was written for
    /// yet is held in memory until [`Self::record_turn`] has something to
    /// store, so every file on disk has a transcript in it. The blank model
    /// spec is [`SessionTrack::turn`]'s to fill, and it runs before any write.
    fn open(dir: StateDir, claim: SessionClaim, stored: Option<StoredSession>, cwd: &str) -> Self {
        let session = stored.unwrap_or_else(|| {
            let mut session = StoredSession::new("", cwd);
            session.id = claim.id();
            session
        });
        Self {
            dir,
            claim,
            session,
        }
    }

    /// `context_size` travels with the messages, since a resumed session seeds
    /// its gauge from it. Stored without one, the next process is back to
    /// estimating a transcript this one had measured.
    ///
    /// An empty transcript is not a session, and this is the one place that
    /// decides so. An empty log is one the picker offers and `--continue`
    /// resolves to, so a run killed before its first turn would leave a dead
    /// entry behind for good. The same guard stops a history that sanitized
    /// down to nothing from replacing the copy it was restored from.
    fn record_turn(
        &mut self,
        messages: &[Message],
        frame: Option<&Arc<StoredFrame>>,
        context_size: u32,
    ) -> Result<(), SessionError> {
        if messages.is_empty() {
            return Ok(());
        }
        self.session.replace_messages(messages.to_vec());
        self.session.set_frame(frame.cloned());
        self.session.meta.context_size = context_size;
        self.session.update_title_if_default();
        self.session.save(&self.claim, &self.dir)
    }
}

// -- `maki.session.messages` payload --

const CURRENT_WINDOW_ID: &str = "current";
const FALLBACK_SUBAGENT_NAME: &str = "subagent";
const UNRESOLVED_TOOL_RESULT: &str = "[tool result unavailable]";

/// The transcript a plugin reads through `maki.session.messages`: the live
/// window first, then every archived pre-compaction window oldest first.
/// Each window carries its messages in the content-block format, the
/// subagent transcripts that belong to it, and a `created_at` that is the
/// best record of when the window began: the session's start, or for a
/// post-compaction window the seal time of the newest archive (the
/// compaction that opened it happened right then). Archives report their
/// own seal time. `seal_at` is that newest seal time for callers that did
/// not load the archives, so the current window's `created_at` does not
/// depend on whether they were requested. `trim_in_flight` cuts the trailing
/// tool batch of the current window when the caller knows a turn is running:
/// its calls have no output yet, and the placeholder that would close
/// them reads as a failure. Only settled content shows; the turn joins
/// once it completes.
pub fn messages_json(
    current: &StoredSession,
    archives: &[StoredArchive],
    seal_at: Option<u64>,
    trim_in_flight: bool,
) -> Value {
    let current_created_at = archives
        .last()
        .map_or_else(|| seal_at.unwrap_or(current.created_at), |newest| {
            newest.session.updated_at
        });
    let attach = subagent_attach(current, archives);
    let mut windows = Vec::with_capacity(archives.len() + 1);
    windows.push(window_json(
        CURRENT_WINDOW_ID,
        current_created_at,
        current,
        &attach[archives.len()],
        trim_in_flight,
    ));
    for (archive, attach) in archives.iter().zip(&attach) {
        windows.push(window_json(
            &archive.seq.to_string(),
            archive.session.updated_at,
            &archive.session,
            attach,
            false,
        ));
    }
    json!({ "id": current.id.to_string(), "windows": windows })
}

fn window_json(
    id: &str,
    created_at: u64,
    session: &StoredSession,
    attach: &[&str],
    trim_in_flight: bool,
) -> Value {
    let messages = session.messages();
    let cut =
        messages.len() - usize::from(trim_in_flight && trailing_batch_unresolved(session));
    let settled = &messages[..cut];
    let mut value = json!(settled);
    if let Some(closing) = dangling_results(settled, session) {
        value.as_array_mut().unwrap().push(json!(closing));
    }
    json!({
        "id": id,
        "created_at": created_at,
        "messages": value,
        "subagents": subagents_json(session, attach),
    })
}

/// The trailing message opens a tool batch whose calls have no recorded
/// output. Nothing after it can answer them, so on a live session the batch
/// is still running and only a completed call's output can settle it - the
/// same criterion [`dangling_results`] would flag as an error placeholder.
fn trailing_batch_unresolved(session: &StoredSession) -> bool {
    session.messages().last().is_some_and(|last| {
        last.tool_uses()
            .map(|(id, _, _)| id)
            .any(|id| !session.tool_outputs().contains_key(id))
    })
}

fn tool_use_ids(session: &StoredSession) -> impl Iterator<Item = &str> {
    session
        .messages()
        .iter()
        .flat_map(|m| m.tool_uses().map(|(id, _, _)| id))
}

/// Per window, oldest first with the current window last, the subagent ids
/// whose transcript attaches there. A transcript belongs to the window
/// holding its parent `tool_use`, and when pruning has taken that window
/// it belongs to the oldest window that still holds its records, which a
/// rewrite always leaves somewhere because it carries every record forward.
fn subagent_attach<'a>(
    current: &'a StoredSession,
    archives: &'a [StoredArchive],
) -> Vec<Vec<&'a str>> {
    let sessions: Vec<&StoredSession> = archives
        .iter()
        .map(|a| &a.session)
        .chain(std::iter::once(current))
        .collect();
    let mut parent_at: HashMap<&str, usize> = HashMap::new();
    let mut records_at: HashMap<&str, usize> = HashMap::new();
    for (idx, session) in sessions.iter().enumerate() {
        for id in tool_use_ids(session) {
            parent_at.insert(id, idx);
        }
        for id in session.subagent_messages().keys() {
            records_at.entry(id.as_str()).or_insert(idx);
        }
    }
    let mut attach = vec![Vec::new(); sessions.len()];
    for (id, records_at) in records_at {
        attach[parent_at.get(id).copied().unwrap_or(records_at)].push(id);
    }
    attach
}

/// A user message closing `tool_use` blocks nothing later in the window
/// answers, which a crash mid-turn can leave behind. Results resolve from
/// the window's `out` records; a call with no recorded output gets a clean
/// error placeholder, so no `tool_use_id` dangles.
fn dangling_results(messages: &[Message], session: &StoredSession) -> Option<Message> {
    let answered: Vec<&str> = messages
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|block| match block {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                _ => None,
            })
        })
        .collect();
    let missing: Vec<&str> = messages
        .iter()
        .flat_map(|m| m.tool_uses().map(|(id, _, _)| id))
        .filter(|id| !answered.contains(id))
        .collect();
    if missing.is_empty() {
        return None;
    }
    let content = missing
        .into_iter()
        .map(|id| match session.tool_outputs().get(id) {
            Some(output) => ContentBlock::ToolResult {
                tool_use_id: id.to_owned(),
                content: output.as_text(),
                is_error: false,
                loaded_tools: output.loaded_tools().to_vec(),
            },
            None => ContentBlock::ToolResult {
                tool_use_id: id.to_owned(),
                content: UNRESOLVED_TOOL_RESULT.into(),
                is_error: true,
                loaded_tools: Vec::new(),
            },
        })
        .collect();
    Some(Message {
        role: Role::User,
        content,
        ..Default::default()
    })
}

/// The subagent transcripts that attach to this window (see
/// [`subagent_attach`]). Names come from the session's `subagents` list; a
/// record with none gets a placeholder.
/// Ordered by spawn with a stable `nr`: the 1-based position in the
/// session's subagent list, assigned at spawn and never reused, so a
/// transcript that attaches after one spawned later does not renumber it.
/// Attached ids the list predates go last, id-sorted.
fn subagents_json(session: &StoredSession, attach: &[&str]) -> Value {
    let meta = session.subagents();
    let mut entries: Vec<(usize, String, &Arc<Vec<Message>>)> = meta
        .iter()
        .enumerate()
        .filter(|(_, sa)| attach.contains(&sa.tool_use_id.as_str()))
        .filter_map(|(i, sa)| {
            Some((
                i + 1,
                sa.name.clone(),
                session.subagent_messages().get(&sa.tool_use_id)?,
            ))
        })
        .collect();
    let mut unlisted: Vec<&str> = attach
        .iter()
        .copied()
        .filter(|id| !meta.iter().any(|sa| sa.tool_use_id == *id))
        .collect();
    unlisted.sort_unstable();
    for (i, id) in unlisted.into_iter().enumerate() {
        if let Some(messages) = session.subagent_messages().get(id) {
            entries.push((meta.len() + i + 1, FALLBACK_SUBAGENT_NAME.to_owned(), messages));
        }
    }
    entries
        .into_iter()
        .map(|(nr, name, messages)| {
            json!({ "nr": nr, "name": name, "messages": messages.as_ref().as_slice() })
        })
        .collect::<Vec<_>>()
        .into()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use maki_providers::{ContentBlock, Role};
    use maki_storage::frame::PromptFacts;
    use maki_storage::id::MakiId;
    use maki_storage::sessions::{SESSIONS_DIR, StoredSubagent, generate_title};
    use tempfile::TempDir;
    use test_case::test_case;

    use crate::RunContext;
    use crate::tools::RequestTools;

    use super::*;
    use crate::agent::live_history;

    const SESSION_ID: &str = "01965087-4c71-7f00-8000-000000000000";
    const CWD: &str = "/project";
    const MODEL_SPEC: &str = "anthropic/claude-test";
    const OTHER_SPEC: &str = "other/model";
    const CONTEXT_SIZE: u32 = 42_000;
    const PROMPT: &str = "fix the login bug";
    const OBSERVATION: &str = "build failed";
    const TITLE: &str = "a title the user set";
    const PLAN_PATH: &str = "/plans/plan.md";
    const NO_EMPTY_FILE: &str = "a session with no transcript must not be on disk";
    const NO_EMPTY_LATEST: &str = "an empty session must not be what --continue resolves to";
    const NOT_WIPED: &str = "an empty history must not replace the transcript it came from";
    const UNTOUCHED: &str = "a run that changed nothing must not rewrite the log";
    const RETRIED: &str = "a turn a failed write dropped has to reach disk on the next one";
    const REPAIRED: &str = "a turn that ran on the repaired transcript has to store it repaired";
    const ORPHAN_TOOL_ID: &str = "tool-nobody-called";
    const ORPHAN_RESULT: &str = "ok";
    const FRAME_SYSTEM: &str = "system the session was sent under";
    const FRAME_FINGERPRINT: &str = "0123456789abcdef";
    const FRAME_MCP_TOOL: &str = "srv__tool";
    const FRAME_KEPT: &str = "a resumed process must send the prefix the transcript was sent under";

    fn session_id() -> MakiId {
        SESSION_ID.parse().unwrap()
    }

    fn log_path(tmp: &TempDir) -> PathBuf {
        tmp.path()
            .join(SESSIONS_DIR)
            .join(format!("{}.jsonl", session_id()))
    }

    /// Stores a prompt followed by a tool result with no call in front of it,
    /// which is what a transcript cut off mid-turn leaves behind, and reopens
    /// it. [`History::restored`] drops the orphan, so the reopened run holds a
    /// repaired copy that differs from the file.
    fn reopen_with_orphan(tmp: &TempDir) -> SessionTrack {
        let orphan = Message {
            role: Role::User,
            content: vec![ContentBlock::tool_result(
                ORPHAN_TOOL_ID,
                ORPHAN_RESULT,
                false,
            )],
            ..Default::default()
        };
        push_turn(&mut track_on(tmp), MODEL_SPEC, |params| {
            params.history.push(Message::user(PROMPT.into()));
            params.history.push(orphan);
        });
        SessionTrack::open(
            Resumed::stored(SessionRef::from(session_id()), load(tmp)),
            claim(tmp),
            state_dir(tmp),
            CWD,
        )
    }

    fn state_dir(tmp: &TempDir) -> StateDir {
        StateDir::from_path(tmp.path().to_path_buf())
    }

    fn load(tmp: &TempDir) -> StoredSession {
        StoredSession::load(session_id(), &state_dir(tmp)).unwrap()
    }

    fn resumed() -> Resumed {
        Resumed::empty(SessionRef::from(session_id()))
    }

    fn claim(tmp: &TempDir) -> SessionClaim {
        SessionClaim::acquire(session_id(), &state_dir(tmp)).expect("nothing else holds it")
    }

    fn track_on(tmp: &TempDir) -> SessionTrack {
        SessionTrack::open(resumed(), claim(tmp), state_dir(tmp), CWD)
    }

    fn push_turn(track: &mut SessionTrack, spec: &str, edit: impl FnOnce(AgentRunParams<'_>)) {
        let mut turn = track.turn(spec.to_owned());
        edit(turn.run_params(
            EventSender::new(flume::unbounded().0, 0),
            Arc::new(|_, _| RunContext::fixed(String::new(), RequestTools::default())),
        ));
    }

    fn push_prompt(track: &mut SessionTrack, spec: &str, text: &str) {
        push_turn(track, spec, |params| {
            params.history.push(Message::user(text.into()))
        });
    }

    /// A run killed before its first turn has to leave nothing behind, neither
    /// a file under its id nor something for `--continue` to land on. `maki -p`
    /// is the reason: it is short, scripted and interrupted often, and the
    /// previous session in the directory has to stay the one `-c` finds.
    #[test_case(false ; "a run that never took a turn")]
    #[test_case(true ; "a turn that produced no messages")]
    fn a_run_with_no_transcript_leaves_nothing_behind(took_turn: bool) {
        let tmp = TempDir::new().unwrap();
        let mut track = track_on(&tmp);
        if took_turn {
            push_turn(&mut track, MODEL_SPEC, |_| {});
        }
        drop(track);

        assert!(
            StoredSession::load(session_id(), &state_dir(&tmp)).is_err(),
            "{NO_EMPTY_FILE}"
        );
        assert!(
            StoredSession::claim_latest(CWD, &state_dir(&tmp))
                .expect("an empty store is readable")
                .is_none(),
            "{NO_EMPTY_LATEST}"
        );
    }

    /// The same guard from the other side: once a transcript is stored, a turn
    /// that ends with an empty history leaves it alone instead of emptying the
    /// only copy of it.
    #[test]
    fn an_empty_history_does_not_wipe_a_stored_transcript() {
        let tmp = TempDir::new().unwrap();
        let mut track = track_on(&tmp);
        push_prompt(&mut track, MODEL_SPEC, PROMPT);
        push_turn(&mut track, MODEL_SPEC, |params| params.history.truncate(0));

        assert_eq!(load(&tmp).messages().len(), 1, "{NOT_WIPED}");
    }

    /// What a driver pushes through `run_params` is on disk once the turn ends,
    /// under the id, cwd, spec, title and measured size a resumed run reads
    /// back. Nothing here calls a save, ending the turn is the save.
    ///
    /// The observation is in there because it is the message kind a transcript
    /// can lose silently: it is not part of the model's reply, so nothing
    /// downstream complains when it fails to round trip.
    #[test]
    fn a_finished_turn_round_trips_through_disk() {
        let tmp = TempDir::new().unwrap();
        let mut track = track_on(&tmp);
        let messages = vec![
            Message::user(PROMPT.into()),
            Message::observation(OBSERVATION.into()),
        ];
        push_turn(&mut track, MODEL_SPEC, |params| {
            for message in &messages {
                params.history.push(message.clone());
            }
            *params.gauge = ContextGauge::restored(CONTEXT_SIZE);
        });

        let loaded = load(&tmp);
        assert_eq!(loaded.id, session_id());
        assert_eq!(loaded.cwd, CWD);
        assert_eq!(loaded.model, MODEL_SPEC);
        assert_eq!(loaded.title, generate_title(&messages));
        assert_eq!(loaded.messages().len(), 2);
        assert!(loaded.messages()[1].is_observation());
        assert_eq!(
            loaded.meta.context_size, CONTEXT_SIZE,
            "a resumed session seeds its gauge from this, so it has to be stored"
        );
    }

    /// `updated_at` counts whole seconds, so a rewrite in the same second as
    /// the seed could leave the bytes as they were. The orphan closes that
    /// gap: the reopened history has already dropped it, so any rewrite at
    /// all changes the file.
    #[test_case(0 ; "a run that took no turn")]
    #[test_case(1 ; "a turn that changed nothing")]
    fn a_run_that_changed_nothing_leaves_the_file_alone(turns: usize) {
        let tmp = TempDir::new().unwrap();
        let mut track = reopen_with_orphan(&tmp);
        let before = std::fs::read(log_path(&tmp)).unwrap();

        for _ in 0..turns {
            push_turn(&mut track, OTHER_SPEC, |_| {});
        }
        drop(track);

        assert_eq!(
            std::fs::read(log_path(&tmp)).unwrap(),
            before,
            "{UNTOUCHED}"
        );
    }

    #[test]
    fn a_turn_that_changed_something_writes_the_repair_with_it() {
        let tmp = TempDir::new().unwrap();
        push_prompt(&mut reopen_with_orphan(&tmp), OTHER_SPEC, "second");

        let has_tool_result = load(&tmp)
            .messages()
            .iter()
            .flat_map(|m| &m.content)
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }));
        assert!(!has_tool_result, "{REPAIRED}");
    }

    /// The second turn changes nothing, so only the flag the failed write
    /// left set can make it save.
    #[test]
    fn a_failed_save_is_retried_by_the_next_turn() {
        let tmp = TempDir::new().unwrap();
        let mut track = track_on(&tmp);
        // A directory where the rewrite puts its temp file: `File::create`
        // cannot replace it, so the save fails before anything lands.
        let blocker = log_path(&tmp).with_extension("jsonl.tmp");
        std::fs::create_dir(&blocker).unwrap();
        push_prompt(&mut track, MODEL_SPEC, PROMPT);
        assert!(
            StoredSession::load(session_id(), &state_dir(&tmp)).is_err(),
            "the blocked write cannot have landed"
        );

        std::fs::remove_dir(&blocker).unwrap();
        push_turn(&mut track, MODEL_SPEC, |_| {});
        drop(track);

        assert_eq!(load(&tmp).messages().len(), 1, "{RETRIED}");
    }

    /// The next process continues the transcript instead of starting one
    /// beside it, which is the whole point of `-c` and `-r`. The title and plan
    /// are set only in memory, so they reach disk only if the run writes back
    /// into the session it was handed rather than reading the file again.
    #[test]
    fn reopening_resumes_the_stored_session() {
        let tmp = TempDir::new().unwrap();
        push_prompt(&mut track_on(&tmp), MODEL_SPEC, PROMPT);
        let mut stored = load(&tmp);
        stored.set_title(TITLE.to_owned());
        stored.meta.plan_path = Some(PLAN_PATH.to_owned());

        let mut track = SessionTrack::open(
            Resumed::stored(SessionRef::from(session_id()), stored),
            claim(&tmp),
            state_dir(&tmp),
            CWD,
        );
        push_prompt(&mut track, OTHER_SPEC, PROMPT);

        let loaded = load(&tmp);
        assert_eq!(loaded.messages().len(), 2);
        assert_eq!(loaded.model, OTHER_SPEC);
        assert_eq!(loaded.title, TITLE);
        assert_eq!(loaded.meta.plan_path.as_deref(), Some(PLAN_PATH));
    }

    /// A turn that never touches the frame still saves the transcript, and
    /// must save the frame next to it. Otherwise every resume starts a new
    /// prefix and loses the cache and the thinking.
    #[test]
    fn the_frame_survives_a_resume_and_the_next_turn() {
        let tmp = TempDir::new().unwrap();
        let frame = Arc::new(StoredFrame {
            system: FRAME_SYSTEM.into(),
            fingerprint: FRAME_FINGERPRINT.into(),
            facts: Some(PromptFacts {
                cwd: CWD.into(),
                ..Default::default()
            }),
            mcp_tools: vec![serde_json::json!({ "name": FRAME_MCP_TOOL })],
        });
        let resumed = Resumed {
            history: vec![Message::user(PROMPT.into())],
            frame: Some(Arc::clone(&frame)),
            ..resumed()
        };
        let mut track = SessionTrack::open(resumed, claim(&tmp), state_dir(&tmp), CWD);
        push_prompt(&mut track, MODEL_SPEC, OBSERVATION);
        drop(track);

        let stored = load(&tmp);
        assert_eq!(stored.frame(), Some(&frame), "{FRAME_KEPT}");
        let reopened = SessionTrack::open(
            Resumed::stored(SessionRef::from(session_id()), stored),
            claim(&tmp),
            state_dir(&tmp),
            CWD,
        );
        assert_eq!(reopened.history.frame(), Some(&frame), "{FRAME_KEPT}");
    }

    /// `maki.session.messages` under a headless driver reads what the track
    /// publishes, so the restored transcript and every turn after it have to
    /// be there, and gone once the run ends. The id is minted so no other test
    /// holding the shared one can publish over it.
    #[test]
    fn open_publishes_the_live_transcript_until_dropped() {
        let tmp = TempDir::new().unwrap();
        let id = MakiId::generate();
        let resumed = Resumed {
            history: vec![Message::user(PROMPT.into())],
            ..Resumed::empty(SessionRef::from(id))
        };
        let claim = SessionClaim::acquire(id, &state_dir(&tmp)).expect("nothing else holds it");
        let mut track = SessionTrack::open(resumed, claim, state_dir(&tmp), CWD);

        let restored = live_history(id).expect("an open track is live");
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].user_text(), Some(PROMPT));

        push_prompt(&mut track, MODEL_SPEC, OBSERVATION);
        let pushed = live_history(id).expect("an open track is live");
        assert_eq!(pushed.len(), 2);
        assert_eq!(pushed[1].user_text(), Some(OBSERVATION));

        drop(track);
        assert!(live_history(id).is_none());
    }

    // -- messages_json --

    const TOOL_ID: &str = "tool-1";
    const SUB_ID: &str = "sub-1";
    const STALE_SUB_ID: &str = "sub-archived";
    const RESOLVED_RESULT: &str = "the recorded output";
    const CURRENT_FIRST: &str = "current window answers first";
    const ARCHIVE_FIRST: &str = "archives answer oldest first";

    fn assistant_tool_use(id: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::tool_use(id, "bash", serde_json::json!({}))],
            ..Default::default()
        }
    }

    fn archive_window(seq: u64, text: &str, updated_at: u64) -> StoredArchive {
        let mut session = StoredSession::new(MODEL_SPEC, CWD);
        session.push_message(Message::user(text.into()));
        session.updated_at = updated_at;
        StoredArchive { seq, session }
    }

    #[test]
    fn messages_json_lists_current_first_then_archives_oldest_first() {
        let mut session = StoredSession::new(MODEL_SPEC, CWD);
        session.push_message(Message::user(PROMPT.into()));
        let archives = [archive_window(1, "old", 100), archive_window(3, "new", 200)];

        let value = messages_json(&session, &archives, None, false);
        let ids: Vec<&str> = value["windows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, [CURRENT_WINDOW_ID, "1", "3"], "{CURRENT_FIRST}");
        let created: Vec<u64> = value["windows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w["created_at"].as_u64().unwrap())
            .collect();
        assert_eq!(created, [200, 100, 200], "{ARCHIVE_FIRST}");
    }

    #[test]
    fn messages_json_without_archives_reports_the_session_start() {
        let mut session = StoredSession::new(MODEL_SPEC, CWD);
        session.push_message(Message::user(PROMPT.into()));

        let value = messages_json(&session, &[], None, false);
        assert_eq!(
            value["windows"][0]["created_at"].as_u64(),
            Some(session.created_at)
        );
    }

    #[test]
    fn messages_json_dates_an_unloaded_current_window_from_the_seal_time() {
        let mut session = StoredSession::new(MODEL_SPEC, CWD);
        session.push_message(Message::user(PROMPT.into()));
        let archives = [archive_window(2, "new", 200)];

        let value = messages_json(&session, &[], Some(500), false);
        assert_eq!(value["windows"][0]["created_at"].as_u64(), Some(500));

        let value = messages_json(&session, &archives, Some(500), false);
        assert_eq!(
            value["windows"][0]["created_at"].as_u64(),
            Some(200),
            "a loaded newest archive outranks the fallback"
        );
    }

    #[test]
    fn messages_json_closes_dangling_tool_uses_from_out_records() {
        let mut session = StoredSession::new(MODEL_SPEC, CWD);
        session.push_message(assistant_tool_use(TOOL_ID));
        session.insert_tool_output(
            TOOL_ID.into(),
            Arc::new(ToolOutput::Plain(RESOLVED_RESULT.into())),
        );

        let window = &messages_json(&session, &[], None, false)["windows"][0];
        let messages = window["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2, "{CURRENT_FIRST}");
        assert_eq!(
            messages[1]["content"][0]["content"].as_str(),
            Some(RESOLVED_RESULT)
        );
    }

    #[test]
    fn messages_json_flags_a_dangling_tool_use_with_no_output() {
        let mut session = StoredSession::new(MODEL_SPEC, CWD);
        session.push_message(assistant_tool_use(TOOL_ID));

        let window = &messages_json(&session, &[], None, false)["windows"][0];
        let block = &window["messages"].as_array().unwrap()[1]["content"][0];
        assert_eq!(block["content"].as_str(), Some(UNRESOLVED_TOOL_RESULT));
        assert_eq!(block["is_error"].as_bool(), Some(true));
    }

    #[test]
    fn messages_json_trims_a_trailing_batch_without_output() {
        let mut session = StoredSession::new(MODEL_SPEC, CWD);
        session.push_message(Message::user(PROMPT.into()));
        session.push_message(assistant_tool_use(TOOL_ID));

        let window = &messages_json(&session, &[], None, true)["windows"][0];
        let messages = window["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1, "only the prompt is settled");
        assert_eq!(messages[0]["role"].as_str(), Some("user"));
    }

    #[test]
    fn messages_json_keeps_a_trailing_batch_whose_outputs_recorded() {
        let mut session = StoredSession::new(MODEL_SPEC, CWD);
        session.push_message(assistant_tool_use(TOOL_ID));
        session.insert_tool_output(
            TOOL_ID.into(),
            Arc::new(ToolOutput::Plain(RESOLVED_RESULT.into())),
        );

        let window = &messages_json(&session, &[], None, true)["windows"][0];
        let messages = window["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2, "a settled batch stays");
        assert_eq!(
            messages[1]["content"][0]["content"].as_str(),
            Some(RESOLVED_RESULT)
        );
        assert!(
            messages[1]["content"][0]["is_error"].is_null(),
            "a settled batch closes without an error flag"
        );
    }

    #[test]
    fn messages_json_keeps_dangling_placeholders_mid_window_when_trimming() {
        let mut session = StoredSession::new(MODEL_SPEC, CWD);
        session.push_message(assistant_tool_use(TOOL_ID));
        session.push_message(Message::user("the turn moved on".into()));

        let window = &messages_json(&session, &[], None, true)["windows"][0];
        let messages = window["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3, "crash residue keeps its placeholder");
        assert_eq!(
            messages[2]["content"][0]["content"].as_str(),
            Some(UNRESOLVED_TOOL_RESULT)
        );
    }

    #[test]
    fn messages_json_attaches_subagents_to_the_window_that_spawned_them() {
        let mut archived = StoredSession::new(MODEL_SPEC, CWD);
        archived.push_message(assistant_tool_use(STALE_SUB_ID));
        archived.set_subagent_messages(
            STALE_SUB_ID.into(),
            vec![Message::user("archived turn".into())],
        );
        archived.set_subagents(vec![StoredSubagent {
            tool_use_id: STALE_SUB_ID.into(),
            name: "scout".into(),
            model: None,
            thinking: None,
            fast: false,
        }]);
        let archives = [StoredArchive {
            seq: 1,
            session: archived,
        }];

        let mut session = StoredSession::new(MODEL_SPEC, CWD);
        session.push_message(assistant_tool_use(SUB_ID));
        session.set_subagent_messages(SUB_ID.into(), vec![Message::user("sub turn".into())]);
        // The rewrite that opened the current window carried the archive's
        // records along, so both windows hold them.
        session.set_subagent_messages(
            STALE_SUB_ID.into(),
            vec![Message::user("archived turn".into())],
        );
        session.set_subagents(vec![StoredSubagent {
            tool_use_id: SUB_ID.into(),
            name: "researcher".into(),
            model: None,
            thinking: None,
            fast: false,
        }]);

        let value = messages_json(&session, &archives, None, false);
        let windows = value["windows"].as_array().unwrap();
        let names = |window: &Value| -> Vec<String> {
            window["subagents"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s["name"].as_str().unwrap().to_owned())
                .collect()
        };
        assert_eq!(names(&windows[0]), ["researcher"], "{CURRENT_FIRST}");
        assert_eq!(names(&windows[1]), ["scout"], "{ARCHIVE_FIRST}");
    }

    /// Pruning drops the oldest archives, so a transcript can outlive the
    /// window that spawned it. Its records were carried into every later
    /// window, so the oldest survivor renders them instead of the
    /// transcript quietly disappearing with its parent.
    #[test]
    fn messages_json_keeps_a_subagent_whose_parent_window_was_pruned() {
        let mut archived = StoredSession::new(MODEL_SPEC, CWD);
        archived.set_subagent_messages(
            STALE_SUB_ID.into(),
            vec![Message::user("orphaned turn".into())],
        );
        let archives = [StoredArchive {
            seq: 2,
            session: archived,
        }];

        let mut session = StoredSession::new(MODEL_SPEC, CWD);
        session.set_subagent_messages(
            STALE_SUB_ID.into(),
            vec![Message::user("orphaned turn".into())],
        );

        let value = messages_json(&session, &archives, None, false);
        let windows = value["windows"].as_array().unwrap();
        assert_eq!(
            windows[0]["subagents"].as_array().unwrap().len(),
            0,
            "{CURRENT_FIRST}"
        );
        let subagents = windows[1]["subagents"].as_array().unwrap();
        assert_eq!(subagents.len(), 1, "{ARCHIVE_FIRST}");
        assert_eq!(subagents[0]["name"].as_str(), Some(FALLBACK_SUBAGENT_NAME));
        assert_eq!(
            subagents[0]["messages"][0]["content"][0]["text"],
            "orphaned turn"
        );
    }

    #[test]
    fn messages_json_names_an_unrecorded_subagent_with_a_placeholder() {
        let mut session = StoredSession::new(MODEL_SPEC, CWD);
        session.push_message(assistant_tool_use(SUB_ID));
        session.set_subagent_messages(SUB_ID.into(), vec![Message::user("sub turn".into())]);

        let window = &messages_json(&session, &[], None, false)["windows"][0];
        assert_eq!(
            window["subagents"][0]["name"].as_str(),
            Some(FALLBACK_SUBAGENT_NAME)
        );
    }

    /// The list order is spawn order with each subagent at the position it
    /// was spawned at, so a consumer can number a transcript's items off
    /// `nr` without a later-attaching earlier spawn renumbering anything.
    #[test]
    fn messages_json_numbers_subagents_by_spawn_position() {
        let mut session = StoredSession::new(MODEL_SPEC, CWD);
        for id in ["sub-1", "sub-2", "sub-unlisted"] {
            session.set_subagent_messages(id.into(), vec![Message::user("sub turn".into())]);
        }
        session.set_subagents(vec![
            StoredSubagent {
                tool_use_id: "sub-1".into(),
                name: "scout".into(),
                model: None,
                thinking: None,
                fast: false,
            },
            StoredSubagent {
                tool_use_id: "sub-2".into(),
                name: "researcher".into(),
                model: None,
                thinking: None,
                fast: false,
            },
        ]);

        let window = &messages_json(&session, &[], None, false)["windows"][0];
        let subagents = window["subagents"].as_array().unwrap();
        assert_eq!(subagents.len(), 3, "every attached transcript lists");
        assert_eq!(subagents[0]["nr"].as_u64(), Some(1));
        assert_eq!(subagents[0]["name"].as_str(), Some("scout"));
        assert_eq!(subagents[1]["nr"].as_u64(), Some(2));
        assert_eq!(subagents[1]["name"].as_str(), Some("researcher"));
        assert_eq!(
            subagents[2]["nr"].as_u64(),
            Some(3),
            "an id the subagent list predates numbers after the listed ones"
        );
        assert_eq!(subagents[2]["name"].as_str(), Some(FALLBACK_SUBAGENT_NAME));
    }
}
