# LESSONS-RS.md - Rust Patterns for BPE Implementation

## Rust Concepts from the Tutorial

### Data Structures and Performance

The tutorial demonstrates how data structure choice impacts performance:
- **HashMap**: For tracking symbol pairs and their frequencies
- **BinaryHeap**: Priority queue for tracking most frequent pairs
- **BTreeMap vs Linked List**: Different approaches to tracking symbol positions

### Key Struct Design

```rust
pub struct Symbol {
    pub start_byte: usize,
    pub end_byte: usize,
    pub prev: isize,      // -1 for no previous
    pub next: isize,      // -1 for no next
    pub size: usize,      // Length in bytes
}
```

### Trait Implementation

```rust
pub trait BpeTokenizer {
    fn tokenize<'a>(&self, input_text: &'a str) -> Vec<&'a str>;
}
```

### Ownership and Lifetimes
- Using lifetime annotations to return string slices from input
- Avoiding allocations by working with indices and slices
- Managing mutable state during the merge process

## Implementation Patterns

### Symbol Tracking
- Using indices (isize) instead of pointers for prev/next relationships
- -1 as sentinel value for null links
- Byte offsets for UTF-8 safety

### Frequency Counting
- Iterating through adjacent pairs
- Updating counts in HashMap
- Finding maximum efficiently

### Merge Operations
1. Identify symbols to merge
2. Update the first symbol's end position
3. Update linking to skip the second symbol
4. Maintain data structure consistency

## Complexity Analysis

The tutorial explores different algorithmic approaches:

1. **Naive**: O(N²)
   - Simple nested loops
   - Good for understanding, poor for production

2. **Pre-split**: O(N)
   - Split by whitespace first
   - Fast but limits functionality

3. **Priority Queue + BST**: O(N log N)
   - Better asymptotic complexity
   - More complex implementation

4. **Priority Queue + Linked List**: O(N log N)
   - Best practical performance
   - Efficient symbol manipulation

## Performance Considerations

### Memory Layout
- Contiguous storage for cache efficiency
- Minimize allocations in hot loops
- Use capacity hints when possible

### Benchmarking Insights
From the tutorial's benchmarks:
- Linked list approach performed best
- Constant factors matter as much as big-O
- Real-world data differs from theoretical analysis

## Common Patterns to Avoid

### String Manipulation
- Don't create new strings for each merge
- Work with indices and slices
- Only materialize strings at the end

### Iterator Invalidation
- Be careful when modifying while iterating
- Use indices instead of iterators when mutating

## Testing Approach

The tutorial likely uses:
- Unit tests for individual functions
- Integration tests with known inputs/outputs
- Benchmarks to compare implementations