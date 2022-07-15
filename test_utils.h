#pragma once
#include <cassert>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

constexpr double err = 1e-8;

// Parse a line of comma separated values into a vector
inline vector<double> parse_csv(string line) {
  stringstream ss(line);
  string substr;
  vector<double> arr;
  while (ss.good()) {
    getline(ss, substr, ',');
    arr.push_back(stod(substr));
  }
  return arr;
}

// Compare two vectors
template <class T> bool cmp_vec(const vector<T> &v1, const vector<T> &v2) {
  if (v1.size() == v2.size()) {
    for (auto i = 0; i < v1.size(); ++i)
      if (abs(v1[i] - v2[i]) > err) {
        printf("Vectors are different: v1[%d]=%f, v2[%d]=%f", i, v1[i], i,
               v2[i]);
        return false;
      }
    cout << "Vectors are equal.\n";
    return true;
  } else {
    cout << "Vectors are not the same size. size(v1)=" << v1.size()
         << " size(v2)=" << v2.size() << "\n";
  }
  return false;
}

struct TestCase {
  vector<double> v1; // Vector 1
  vector<double> v2; // Vector 2
  vector<double> gt; // Ground truth

  TestCase(string fname) {
    ifstream fs(fname);
    string line;
    try {

      getline(fs, line); // v1
      v1 = parse_csv(line);
      getline(fs, line); // v2
      v2 = parse_csv(line);
      getline(fs, line); // gt
      gt = parse_csv(line);

    } catch (exception &e) {
      cout << "Failed to read TestCase " << fname << endl;
      cout << e.what() << endl;
    }
  }
};
