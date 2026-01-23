# Changelog

All notable changes to XERV Crayon will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-23

### Added
- **Double-Array Trie (DAT) Engine**: Complete rewrite of the tokenization engine using memory-mapped DAT for O(1) lookups
- **AVX2/SIMD Optimizations**: Native C++ engine with AVX2 intrinsics achieving >16M tokens/second
- **Pre-built Vocabulary Profiles**: 5 production-ready profiles (lite, code, science, multilingual, arts_commerce)
- **CLI Tool**: `crayon-benchmark` command for easy performance testing
- **Zero-Copy Memory Mapping**: Memory-mapped DAT files for instant loading
- **Cross-Platform Support**: Windows (MSVC), Linux (GCC), macOS (Clang/Apple Silicon)

### Changed
- Version bump from 1.1.0 to 2.0.0
- Minimum Python version updated to 3.10
- Package structure reorganized for better modularity

### Performance
- Tokenization: 16M+ tokens/second (up from 2M in v1.x)
- Memory usage: 50% reduction via mmap
- Load time: <10ms for vocabulary profiles

## [1.1.0] - 2026-01-16

### Added
- Initial C-Trie implementation
- SIMD-accelerated text processing
- Basic vocabulary management

### Fixed
- Memory leaks in trie traversal
- Unicode handling edge cases

## [1.0.0] - 2026-01-11

### Added
- Initial release
- Pure Python tokenizer
- Basic vocabulary training
- Entropy-guided vocabulary construction
