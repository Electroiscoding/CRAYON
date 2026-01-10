#ifndef TRIE_NODE_H
#define TRIE_NODE_H

#include <stdint.h>

// 64-byte aligned `TrieNode` struct
typedef struct __attribute__((aligned(64))) TrieNode {
    int32_t id;
    int32_t children_offset;
    // ... other fields
} TrieNode;

#endif
