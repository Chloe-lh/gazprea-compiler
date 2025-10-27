#pragma once

#include <string>
#include <vector>
#include <variant>

// Type enum for all nodes
enum class ValueType {
    UNKNOWN,
    BOOL,
    CHARACTER,
    INTEGER,
    VECTOR,
    REAL,
    TUPLE,
    STRING,
    STRUCT
};

std::string toString(ValueType type);

inline ValueType promote(ValueType from, ValueType to) {
    switch (from) {
        case ValueType::UNKNOWN:
            return ValueType::UNKNOWN;
            break;
        case ValueType::BOOL:
            switch(to) {
                case ValueType::BOOL:       return ValueType::BOOL;
                case ValueType::VECTOR:     return ValueType::VECTOR; // implicit
            }
            break;
        case ValueType::INTEGER:
            switch (to) {
                case ValueType::INTEGER:    return ValueType::INTEGER; 
                case ValueType::VECTOR:     return ValueType::VECTOR;   // implicit
                case ValueType::REAL:       return ValueType::REAL;
            }
            break;
        case ValueType::VECTOR:
            switch (to) {
                case ValueType::VECTOR:     return ValueType::VECTOR;
            }
            break;
        case ValueType::REAL:
            switch (to) {
                case ValueType::REAL:       return ValueType::REAL;
            }
            break;
        case ValueType::TUPLE:
            switch (to) {
                case ValueType::TUPLE:      return ValueType::TUPLE;    // only if same len + internal types are convertible element-wise
            }
            break;
    }
    return ValueType::UNKNOWN;
}
