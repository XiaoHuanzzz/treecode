#ifndef TREECODE_INTERACTION_INDICES_HPP
#define TREECODE_INTERACTION_INDICES_HPP
#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace treecode_list {
class IndexView {
  inline static const int empty_ = 0;
  const int* data_ = &empty_;
  std::size_t size_ = 0;
 public:
  IndexView() = default;
  IndexView(const int* data, std::size_t size) : data_(size ? data : &empty_), size_(size) {}
  explicit IndexView(const std::vector<int>& values) : IndexView(values.data(), values.size()) {}
  const int* begin() const { return data_; }
  const int* end() const { return data_ + size_; }
  std::size_t size() const { return size_; }
  const int& operator[](std::size_t i) const { return data_[i]; }
};

struct IndexBlock {
  // Uninitialized: the owning worker first-touches only its own lists.
  std::unique_ptr<int[]> values;
  explicit IndexBlock(std::size_t size) : values(new int[size]) {}
};

// Owning, immutable ranges after construction. Lists retain their blocks after
// worker destruction. Copies share READ-ONLY storage; clearing or reassigning
// one list cannot change the entries of another list.
class Indices {
  std::vector<int> owned_;
  std::shared_ptr<IndexBlock> block_;
  std::size_t offset_ = 0, count_ = 0;
 public:
  IndexView view() const {
    return block_ ? IndexView(block_->values.get() + offset_, count_) : IndexView(owned_);
  }
  const int* begin() const { return view().begin(); }
  const int* end() const { return view().end(); }
  std::size_t size() const { return view().size(); }
  bool empty() const { return size() == 0; }
  const int& operator[](std::size_t i) const { return view()[i]; }
  void clear() { owned_.clear(); block_.reset(); offset_ = count_ = 0; }
  template<class Iterator> void assign(Iterator first, Iterator last) {
    owned_.assign(first, last);  // Retain a pooled source until the copy finishes.
    block_.reset(); offset_ = count_ = 0;
  }
  Indices& operator=(const std::vector<int>& source) {
    assign(source.begin(), source.end()); return *this;
  }
  std::vector<int> copy() const { return std::vector<int>(begin(), end()); }
  bool operator==(const std::vector<int>& other) const {
    return size() == other.size() && std::equal(begin(), end(), other.begin());
  }
  void set_block(std::shared_ptr<IndexBlock> block, std::size_t offset, std::size_t count) {
    std::vector<int>().swap(owned_);
    block_ = std::move(block); offset_ = offset; count_ = count;
  }
};

// One arena per worker, never shared while writing. A list cannot cross block
// boundaries, preserving contiguous ordered reads. Unused space consists of
// one tail per worker and padding between lists. Oversize lists are supported.
class IndexArena {
  static constexpr std::size_t block_entries = 65536;  // 256 KiB
  std::shared_ptr<IndexBlock> current_;
  std::size_t capacity_ = 0, used_ = 0;
  std::size_t allocated_entries_ = 0, stored_entries_ = 0;
 public:
  void store(Indices& destination, const std::vector<int>& source) {
    if (source.empty()) { destination.clear(); return; }
    if (source.size() > capacity_ - used_) {
      capacity_ = std::max(block_entries, source.size());
      current_ = std::make_shared<IndexBlock>(capacity_);
      allocated_entries_ += capacity_;
      used_ = 0;
    }
    std::copy(source.begin(), source.end(), current_->values.get() + used_);
    destination.set_block(current_, used_, source.size());
    used_ += source.size();
    stored_entries_ += source.size();
  }
  std::size_t allocated_bytes() const { return allocated_entries_ * sizeof(int); }
  std::size_t stored_bytes() const { return stored_entries_ * sizeof(int); }
};
}  // namespace treecode_list
#endif
