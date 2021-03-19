// Reference: https://github.com/K-Adam/SafeQueue

#ifndef __SAFEQUEUE_H__
#define __SAFEQUEUE_H__

#include <mutex>
#include <queue>

template<typename T>
class SafeQueue
{
public:
    SafeQueue() {}

    ~SafeQueue() {}

    void push(const T& obj)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

		m_queue.push(obj);
    }

    bool pop(T& obj)
    {
		std::lock_guard<std::mutex> lock(m_mutex);

		if (m_queue.empty())
		{
			return false;
		}

		obj = m_queue.front();
		m_queue.pop();

		return true;
    }

    size_t size()
    {
		std::lock_guard<std::mutex> lock(m_mutex);

		size_t s = m_queue.size();

        return s;
    }

private:
    std::queue<T> m_queue;
    std::mutex m_mutex;
};


#endif  // __QUEUE_H__
