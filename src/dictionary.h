/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "sent.h"

#include <fastbpe/API.hpp>

#include <istream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "args.h"
#include "real.h"

namespace fasttext {

typedef int32_t id_type;
enum class entry_type : uint8_t {
  word = 1,
  label = 2,
  phrase = 4,
  kbconcept = 8,
  subword = 16,
  all = 255
  };

inline bool contains(entry_type e, entry_type t){
  return (static_cast<uint32_t>(e) & static_cast<uint32_t>(t)) != 0;
}
template<class... EntryTypes>
inline entry_type combine(EntryTypes... rest){
  auto t = ( ... | static_cast<uint32_t>(rest) );
  return entry_type{t};
}

struct entry {
  std::string word;
  int64_t count;
  std::vector<int32_t> subwords;
  std::vector<uint32_t> hashes;
  entry_type type;
  uint8_t pos_tag;
};


class Dictionary {
 protected:
  static const int32_t MAX_VOCAB_SIZE = 150'000'000;
  static const int32_t MAX_LINE_SIZE = 1024;

  int32_t find(const std::string_view&, uint8_t pos_tag = 0) const;
  int32_t find(const std::string_view&, uint32_t h, uint8_t pos_tag = 0) const;
  int32_t find(uint32_t h) const;
  void initTableDiscard();
  void initNgrams();
  void initSubwords();
  std::vector<std::string>
  extractSubwords(const std::string& s)const;
  void reset(std::istream&) const;
  void pushHash(std::vector<int32_t>&, int32_t) const;
  void addSubwords(std::vector<int32_t>&, const std::string&, int32_t) const;

  std::shared_ptr<Args> args_;
  std::vector<int32_t> word2int_;
  std::vector<entry> words_;

  std::vector<real> pdiscard_;
  int32_t size_;
  int32_t nwords_;
  int32_t nlabels_;
  int32_t nsubwords_;
  int32_t nphrases_;
  int32_t nkbconcepts_;
  int64_t ntokens_;

  int64_t pruneidx_size_;

  std::shared_ptr<fastBPE::Encoder> encoder_;

  std::unordered_map<int32_t, int32_t> pruneidx_;
  void addWordNgrams(
      std::vector<int32_t>& line,
      const std::vector<int32_t>& hashes,
      int32_t n) const;

 public:
  static const std::string EOS;
  static const std::string BOW;
  static const std::string EOW;

  explicit Dictionary(std::shared_ptr<Args>);
  explicit Dictionary(std::shared_ptr<Args>, std::istream&);
  int32_t size(entry_type types = entry_type::all) const;
  int32_t nwords() const;
  int32_t nlabels() const;
  int64_t ntokens() const;
  int32_t getId(const std::string&, uint8_t pos_tag = 0) const;
  int32_t getId(const std::string&, uint32_t h) const;
  entry_type getType(int32_t) const;
  entry_type getType(const std::string&) const;
  int getPoS(uint32_t) const;
  bool discard(int32_t, real) const;
  std::string getWord(int32_t) const;
  const std::vector<int32_t>& getSubwords(int32_t) const;
  const std::vector<int32_t> getSubwords(const std::string&, uint8_t pos_tag) const;
  void getSubwords(
      const std::string&,
      std::vector<int32_t>&,
      std::vector<std::string>&) const;
  void computeSubwords(
      const std::string&,
      std::vector<int32_t>&,
      std::vector<std::string>* substrings = nullptr) const;
  uint32_t hash(const std::string_view& str) const;
  uint32_t hash(const std::string_view& str, uint8_t pos_tag) const;
  void add(const std::string&);
  void addWord(const word_t& w);
  void addPhrase(const phrase_t& p, const sent_t::words_array_t& words);
  void addLine(const line_t& line);
  void addSent(const sent_t& sent);
  std::pair<uint32_t, int32_t> addSubword(const std::string&);
  bool readWord(std::istream&, std::string&) const;
  void readFromFile(std::istream&);
  std::string getLabel(int32_t) const;
  void save(std::ostream&) const;
  void load(std::istream&);
  void initSubwordsPos();
  std::vector<int64_t> getCounts(entry_type) const;
  int32_t getLine(std::istream&, std::vector<int32_t>&, std::vector<int32_t>&)
      const;
  int32_t getLine(std::istream&, std::vector<int32_t>&, std::minstd_rand&)
      const;
  int32_t getLine(std::istream&, compact_line_t& line, std::minstd_rand&)
    const;
  void threshold(int64_t, int64_t);
  void prune(std::vector<int32_t>&);
  bool isPruned() {
    return pruneidx_size_ >= 0;
  }
  void dump(std::ostream&) const;
  void init();
};

} // namespace fasttext
