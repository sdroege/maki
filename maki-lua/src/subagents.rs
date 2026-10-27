//! Subagent runs a plugin can talk to while they run, under every frontend.
//!
//! A run registers its inbox for as long as it runs, so the map holds exactly
//! the tasks `maki.task.send` can reach and nothing has to sweep it: the guard
//! a run holds is removed when the run ends, is cancelled, or is dropped.
//! Idles are absent, which is what a caller sending mid-turn wants to know.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};

use maki_agent::{AgentInput, SubagentInbox};

static RUNS: LazyLock<Mutex<HashMap<String, Arc<SubagentInbox>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn lock() -> MutexGuard<'static, HashMap<String, Arc<SubagentInbox>>> {
    RUNS.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One running subagent, retired with the run that opened it.
pub(crate) struct Run {
    task_id: String,
    inbox: Arc<SubagentInbox>,
}

impl Drop for Run {
    fn drop(&mut self) {
        let mut runs = lock();
        // A retried prompt under the same task id registered after us, and
        // retiring that would silence a run that is still going.
        if runs
            .get(&self.task_id)
            .is_some_and(|live| Arc::ptr_eq(live, &self.inbox))
        {
            runs.remove(&self.task_id);
        }
    }
}

pub(crate) struct LiveSubagents;

impl LiveSubagents {
    /// Takes a run's inbox for the length of the returned guard. Several runs
    /// can share a task id; the last one to start owns its messages.
    pub(crate) fn open(task_id: &str, inbox: &Arc<SubagentInbox>) -> Run {
        let run = Run {
            task_id: task_id.to_owned(),
            inbox: Arc::clone(inbox),
        };
        lock().insert(run.task_id.clone(), Arc::clone(&run.inbox));
        run
    }

    /// Queues `input` for a running subagent, which reads it at its next turn
    /// boundary as an interrupt. Errors when no run answers that task id.
    pub(crate) fn send(task_id: &str, input: AgentInput) -> Result<(), String> {
        let inbox = lock()
            .get(task_id)
            .cloned()
            .ok_or_else(|| format!("task is not running: {task_id}"))?;
        inbox.push(input);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use maki_agent::{AgentMode, InputSource};
    use maki_providers::ThinkingConfig;

    use super::*;

    const TASK_ID: &str = "toolu_live";

    fn input(message: &str) -> AgentInput {
        AgentInput {
            message: message.into(),
            mode: AgentMode::Build,
            images: Vec::new(),
            preamble: Vec::new(),
            earlier: Vec::new(),
            thinking: ThinkingConfig::Off,
            fast: false,
            workflow: false,
            prompt: None,
            source: InputSource::Plugin,
        }
    }

    #[test]
    fn send_reaches_a_running_subagent_and_nobody_else() {
        let inbox = Arc::new(SubagentInbox::default());
        assert!(
            LiveSubagents::send(TASK_ID, input("before")).is_err(),
            "a task that never ran must refuse"
        );

        let run = LiveSubagents::open(TASK_ID, &inbox);
        LiveSubagents::send(TASK_ID, input("wrap up")).unwrap();
        assert_eq!(inbox.texts(), ["wrap up"]);

        drop(run);
        assert!(
            LiveSubagents::send(TASK_ID, input("after")).is_err(),
            "a finished run must stop answering"
        );
        assert_eq!(inbox.texts(), ["wrap up"], "the dropped run left nothing");
    }

    #[test]
    fn a_retired_run_leaves_the_same_tasks_newer_run_alone() {
        let first = Arc::new(SubagentInbox::default());
        let second = Arc::new(SubagentInbox::default());
        let old = LiveSubagents::open(TASK_ID, &first);
        let fresh = LiveSubagents::open(TASK_ID, &second);

        drop(old);
        LiveSubagents::send(TASK_ID, input("still here")).unwrap();
        assert_eq!(second.texts(), ["still here"]);
        assert!(first.is_empty());

        drop(fresh);
        assert!(LiveSubagents::send(TASK_ID, input("gone")).is_err());
    }
}
