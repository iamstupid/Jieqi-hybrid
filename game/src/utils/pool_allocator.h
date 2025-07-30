//
// Created by zball on 25-7-23.
// Revised to correctly handle C++ object lifetime.
//

#pragma once
#ifndef VERSION_POOL_ALLOCATOR_H
#define VERSION_POOL_ALLOCATOR_H

#include <memory>
#include <vector>
#include <stack>
#include <algorithm>
#include <utility> // For std::forward

namespace lczero {
    // Anonymous namespace to limit linkage of helper structs to this file.
    namespace _block{
        struct _block {
            _block *next;
            char buffer[0];
        };

        struct _block_deque {
            _block *head = nullptr;
            size_t byte_per_block, byte_per_elem, block_ptr, n_mem_allocated;

            _block_deque(size_t elem_size, size_t n_elem) :
                    byte_per_elem(elem_size),
                    block_ptr(byte_per_block),
                    n_mem_allocated(0){
                byte_per_block = elem_size * n_elem;
                block_ptr = byte_per_block;
            }

            // Destructor correctly cleans up the block chain.
            ~_block_deque() {
                while (head) {
                    auto next = head->next;
                    // The block was allocated as a char array
                    delete[] reinterpret_cast<char*>(head);
                    head = next;
                }
            }

            void alloc_block() {
                // Allocate raw memory for the block struct and the buffer.
                char* raw_mem = new char[sizeof(_block) + byte_per_block];
                _block* new_head = reinterpret_cast<_block*>(raw_mem);

                new_head->next = head;
                head = new_head;
                block_ptr = 0;
                n_mem_allocated += byte_per_block;
            }

            inline void *alloc_elem() {
                if (block_ptr >= byte_per_block) {
                    alloc_block();
                }
                void* result = head->buffer + block_ptr;
                block_ptr += byte_per_elem;
                return result;
            }
        };
    }

    template<typename T, size_t arena_size = 8192>
    class pool_allocator {
    public:
        // Constructor is now public
        pool_allocator() : buffer(sizeof(T), arena_size) {}

        /**
         * @brief Retrieves memory and constructs an object in-place.
         * @tparam Args Constructor argument types.
         * @param args Arguments forwarded to the T's constructor.
         * @return A pointer to the newly constructed object.
         */
        template<typename... Args>
        T* New(Args&&... args) {
            void* mem;
            if (!recycler.empty()) {
                mem = recycler.top();
                recycler.pop();
            } else {
                mem = buffer.alloc_elem();
            }
            // Use placement new to construct the object in the allocated memory
            return new(mem) T(std::forward<Args>(args)...);
        }

        size_t allocated_mem(){ return buffer.n_mem_allocated; }
        size_t recycler_size(){ return recycler.size(); }

        /**
         * @brief Destroys the object and returns its memory to the pool.
         * @param elem A pointer to the object to be destroyed and recycled.
         */
        void Delete(T* elem) {
            if (elem) {
                // Explicitly call the destructor
                elem->~T();
                // Push the raw memory back onto the recycler stack
                recycler.push(elem);
            }
        }

    private:
        using _block_deque = _block::_block_deque;
        _block_deque buffer;
        std::stack<void*> recycler;
    };

    template<typename T>
    class StaticPool {
    public:
        // This function provides safe, lazy, and thread-safe initialization.
        // The static 'instance' is created only on the first call and is
        // automatically destroyed at program termination.
        static lczero::pool_allocator<T, 8192>& get_instance(){
            static lczero::pool_allocator<T, 8192> instance;
            return instance;
        }

        // Static 'New' function to create objects
        template<typename... Args>
        inline static T* New(Args&&... args) {
            return get_instance().New(std::forward<Args>(args)...);
        }

        // Static 'Delete' function
        inline static void Delete(T* p) {
            get_instance().Delete(p);
        }
    };

    template <typename T>
    struct StaticPoolDeleter {
        void operator()(T* p) const {
            StaticPool<T>::Delete(p);
        }
    };
}

#endif //VERSION_POOL_ALLOCATOR_H