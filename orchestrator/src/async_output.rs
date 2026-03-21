//! 异步 I/O 输出引擎。
//!
//! [`AsyncWriter`] 将文件写出任务派送到后台专用线程（`lbm-io`）执行。
//! 主循环只需提取数据、封装闭包、提交任务，立即返回继续推进仿真。
//!
//! ## 背压机制
//! 通道为有界通道（容量由 [`AsyncWriter::new`] 参数指定）。
//! 当积压超过容量时，[`AsyncWriter::submit`] 自动阻塞主线程防止内存无限增长。
//!
//! ## MPI 注意
//! MPI 集合通信操作（MPI_Gatherv 等）不能推入后台线程，需在主线程同步执行。
//! 只有通信完成后得到的数据（Vec<f64>）才被移入异步写出任务。

use std::sync::mpsc;
use std::thread;
use anyhow::Result;

type Task = Box<dyn FnOnce() -> Result<()> + Send + 'static>;

/// 后台 I/O 写出引擎。
pub struct AsyncWriter {
    tx:     mpsc::SyncSender<Option<Task>>,
    handle: Option<thread::JoinHandle<Vec<anyhow::Error>>>,
}

impl AsyncWriter {
    /// 创建后台 I/O 线程，通道容量为 `capacity` 个待处理任务。
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = mpsc::sync_channel::<Option<Task>>(capacity);
        let handle = thread::Builder::new()
            .name("lbm-io".into())
            .spawn(move || {
                let mut errors: Vec<anyhow::Error> = Vec::new();
                while let Ok(msg) = rx.recv() {
                    match msg {
                        Some(task) => { if let Err(e) = task() { errors.push(e); } }
                        None => break,
                    }
                }
                errors
            })
            .expect("Failed to spawn async I/O thread");
        Self { tx, handle: Some(handle) }
    }

    /// 提交写出任务（通道满时阻塞，提供背压）。
    pub fn submit<F>(&self, task: F)
    where F: FnOnce() -> Result<()> + Send + 'static {
        self.tx.send(Some(Box::new(task))).ok();
    }

    /// 等待所有任务完成，返回收集到的错误。
    pub fn shutdown(mut self) -> Vec<anyhow::Error> {
        self.tx.send(None).ok();
        if let Some(h) = self.handle.take() { h.join().unwrap_or_default() } else { Vec::new() }
    }
}

impl Drop for AsyncWriter {
    fn drop(&mut self) {
        self.tx.send(None).ok();
        if let Some(h) = self.handle.take() { let _ = h.join(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_executes_tasks() {
        let counter = Arc::new(Mutex::new(0u32));
        let writer = AsyncWriter::new(4);
        for _ in 0..10 {
            let c = Arc::clone(&counter);
            writer.submit(move || { *c.lock().unwrap() += 1; Ok(()) });
        }
        let errors = writer.shutdown();
        assert!(errors.is_empty());
        assert_eq!(*counter.lock().unwrap(), 10);
    }

    #[test]
    fn test_collects_errors() {
        let writer = AsyncWriter::new(4);
        writer.submit(|| Err(anyhow::anyhow!("test error")));
        let errors = writer.shutdown();
        assert_eq!(errors.len(), 1);
    }
}
