//
// Created by Finn Lukas Busch on 9/20/22.
//

#ifndef RLSIMLIBRARY_THREADPOOL_H
#define RLSIMLIBRARY_THREADPOOL_H
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
/**
 * Generic ThreadPool Class Implementation. Can be used within simulation or outside to execute multiple simulations
 * in parallel (then, an additional interface is required).
 */

class ThreadPool {
private:
    bool should_terminate = false;
    std::mutex queue_mutex;
    std::condition_variable mutex_condition;
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> jobs;
    void ThreadLoop();
public:
    void stop();
    bool busy();
    ThreadPool();
    void QueueJob(const std::function<void()>& job);


};


#endif //RLSIMLIBRARY_THREADPOOL_H
