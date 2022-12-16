//
// Created by finn on 9/20/22.
//

#include "ThreadPool.h"

/**
 * Initialize thread pool
 */
ThreadPool::ThreadPool() {
    const uint32_t num_threads = std::thread::hardware_concurrency();
    threads.resize(num_threads);
    for(uint32_t i = 0; i < num_threads; i++){
        threads.at(i) = std::thread(&ThreadPool::ThreadLoop, this);
    }
}

/**
 * Queue job to thread pool
 * @param job function to queue
 */
void ThreadPool::QueueJob(const std::function<void()> &job) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        jobs.push(job);
    }
    mutex_condition.notify_one();
}

/**
 * idle loop, keeps accepting jobs. Keep alive as long as threadpool is to be used
 */
void ThreadPool::ThreadLoop() {
    while(true){
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            mutex_condition.wait(lock, [this] {
                return !jobs.empty() || should_terminate;

            });
            if(should_terminate){
                return;
            }
            job = jobs.front();
            jobs.pop();
        }
        job();
    }
}

/**
 * Stop thread pool
 */
void ThreadPool::stop() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        should_terminate = true;
    }
    mutex_condition.notify_all();
    for(std::thread& active_thread:threads){
        active_thread.join();
    }
    threads.clear();
}

/**
 * Check if all jobs are done
 * @return bool - if true, it's busy
 */
bool ThreadPool::busy() {
    bool poolbusy;
    {
        std::unique_lock<std::mutex>lock(queue_mutex);
        poolbusy = jobs.empty();
    }
    return !poolbusy;
}
