#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>

namespace nc {
namespace ir {

/// Data type enumeration
enum class DType : uint8_t {
    F32,    // 32-bit float
    F16,    // 16-bit float
    I32,    // 32-bit integer
    I8,     // 8-bit integer
    Bool,   // boolean
};

/// Convert DType to string
inline std::string dtypeToString(DType dtype) {
    switch (dtype) {
        case DType::F32:  return "f32";
        case DType::F16:  return "f16";
        case DType::I32:  return "i32";
        case DType::I8:   return "i8";
        case DType::Bool: return "bool";
    }
    return "unknown";
}

/// Parse string to DType
inline DType stringToDType(const std::string& s) {
    if (s == "f32")  return DType::F32;
    if (s == "f16")  return DType::F16;
    if (s == "i32")  return DType::I32;
    if (s == "i8")   return DType::I8;
    if (s == "bool") return DType::Bool;
    throw std::invalid_argument("Unknown dtype: " + s);
}

/// Get byte size of DType
inline size_t dtypeSize(DType dtype) {
    switch (dtype) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::I32:  return 4;
        case DType::I8:   return 1;
        case DType::Bool: return 1;
    }
    return 0;
}

/// Shape representation (static shape for now)
class Shape {
public:
    Shape() = default;
    Shape(std::initializer_list<int64_t> dims) : dims_(dims) {}
    explicit Shape(std::vector<int64_t> dims) : dims_(std::move(dims)) {}

    /// Number of dimensions
    size_t rank() const { return dims_.size(); }

    /// Get dimension at index (with bounds checking)
    int64_t operator[](size_t i) const {
        if (i >= dims_.size()) {
            throw std::out_of_range("Shape index out of bounds");
        }
        return dims_[i];
    }
    int64_t& operator[](size_t i) {
        if (i >= dims_.size()) {
            throw std::out_of_range("Shape index out of bounds");
        }
        return dims_[i];
    }

    /// Total number of elements
    int64_t numel() const {
        if (dims_.empty()) return 1; // scalar
        int64_t n = 1;
        for (auto d : dims_) n *= d;
        return n;
    }

    /// Access underlying vector
    const std::vector<int64_t>& dims() const { return dims_; }
    std::vector<int64_t>& dims() { return dims_; }

    bool operator==(const Shape& other) const { return dims_ == other.dims_; }
    bool operator!=(const Shape& other) const { return !(*this == other); }

private:
    std::vector<int64_t> dims_;
};

/// Type = DType + Shape
class Type {
public:
    Type() : dtype_(DType::F32) {}
    Type(DType dtype, Shape shape) : dtype_(dtype), shape_(std::move(shape)) {}

    DType dtype() const { return dtype_; }
    const Shape& shape() const { return shape_; }
    Shape& shape() { return shape_; }

    /// Total bytes
    size_t byteSize() const { return shape_.numel() * dtypeSize(dtype_); }

    bool operator==(const Type& other) const {
        return dtype_ == other.dtype_ && shape_ == other.shape_;
    }

private:
    DType dtype_;
    Shape shape_;
};

} // namespace ir
} // namespace nc
