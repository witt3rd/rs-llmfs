# LESSONS-LLM.md - Byte-Pair Encoding

## Overview

Following Guillaume Bé's tutorial on Byte-Pair Encoding (BPE), we explore this sub-word tokenization algorithm used in modern NLP systems. BPE creates a balance between character-level and word-level tokenization by building up a vocabulary through bottom-up aggregation.

## Key Concepts

### What is Byte-Pair Encoding?

BPE is a tokenization method that:
- Starts with individual characters as the base vocabulary
- Iteratively merges the most frequent adjacent character pairs
- Builds sub-word tokens that capture semantic meaning
- Keeps vocabulary size manageable while handling unknown words

### The Core Algorithm

1. Initialize with character-level tokens
2. Count frequency of all adjacent symbol pairs
3. Merge the most frequent pair into a new symbol
4. Update all occurrences of that pair
5. Repeat until target vocabulary size or no more merges

### Why BPE Matters for LLMs

- **Semantic Meaningfulness**: Sub-words often correspond to morphemes (prefixes, roots, suffixes)
- **Vocabulary Efficiency**: Smaller vocabulary than word-level, more meaningful than character-level
- **Handling Unknown Words**: Can tokenize any input by falling back to smaller units
- **Cross-lingual Benefits**: Works across languages without language-specific rules

## Implementation Complexity

The tutorial explores four different approaches with varying time complexities:

1. **Naive**: O(N²) - Simple but inefficient
2. **Pre-split**: O(N) - Fast but limits cross-word merges
3. **Priority Queue + BST**: O(N log N) - Better algorithmic complexity
4. **Priority Queue + Linked List**: O(N log N) - Best practical performance

## Key Insight

"The choice of a data structure can have a significant impact on a real NLP application" - the tutorial demonstrates how algorithmic complexity and implementation details both matter for real-world performance.