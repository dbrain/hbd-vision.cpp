#pragma once

#include <cctype>
#include <cstdlib>
#include <string>

namespace visp {

inline bool env_flag(char const* name, bool default_value = false) {
    char const* v = std::getenv(name);
    if (!v) {
        return default_value;
    }
    std::string s;
    for (char const* p = v; *p; ++p) {
        s.push_back((char)std::tolower((unsigned char)*p));
    }
    if (s.empty() || s == "0" || s == "false" || s == "no" || s == "off") {
        return false;
    }
    if (s == "1" || s == "true" || s == "yes" || s == "on") {
        return true;
    }
    return default_value;
}

} // namespace visp
