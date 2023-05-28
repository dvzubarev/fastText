/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dictionary.h"
#include "sent.hxx"

#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/transformed.hpp>

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace fasttext {

const std::string Dictionary::EOS = "</s>";
const std::string Dictionary::BOW = "<";
const std::string Dictionary::EOW = ">";

Dictionary::Dictionary(std::shared_ptr<Args> args)
    : args_(args),
      word2int_(MAX_VOCAB_SIZE, -1),
      size_(0),
      nwords_(0),
      nlabels_(0),
      nsubwords_(0),
      nphrases_(0),
      nkbconcepts_(0),
      ntokens_(0),
      pruneidx_size_(-1){
  encoder_.reset(new fastBPE::Encoder(args->bpeCodesPath, false));
}

Dictionary::Dictionary(std::shared_ptr<Args> args, std::istream& in)
    : args_(args),
      size_(0),
      nwords_(0),
      nlabels_(0),
      nsubwords_(0),
      nphrases_(0),
      nkbconcepts_(0),
      ntokens_(0),
      pruneidx_size_(-1){
  encoder_.reset(new fastBPE::Encoder());
  load(in);
}

int32_t Dictionary::find(const std::string_view& w, uint8_t pos_tag, entry_type et) const {
  return find(w, hash(w, pos_tag), pos_tag, et);
}

int32_t Dictionary::find(const std::string_view& str, uint32_t h,
                         uint8_t pos_tag, const entry_type et) const {
  int32_t word2intsize = word2int_.size();
  int32_t id = h % word2intsize;
  int32_t pos = word2int_[id];
  while (pos != -1 ) {
    const auto& w = words_[pos];
    if (!str.empty()) {
      if (w.word == str && w.pos_tag == pos_tag && contains(et, w.type))
        break;
    }  else if(hash(w.word, w.pos_tag) == h && contains(et, w.type))
      break;

    id = (id + 1) % word2intsize;
    pos = word2int_[id];
  }
  return id;
}

int32_t Dictionary::find(uint32_t h, entry_type et) const {
  static std::string empty;
  return find(empty, h, 0, et);
}

void Dictionary::add(const std::string& w) {
  int32_t h = find(w);
  ntokens_++;
  if (word2int_[h] == -1) {
    entry e;
    e.word = w;
    e.count = 1;
    e.type = getType(w);
    words_.push_back(e);
    word2int_[h] = size_++;
  } else {
    words_[word2int_[h]].count++;
  }
}

void Dictionary::addConcepts(const std::vector<std::string>& v){
  for (const std::string& s : v)
    addConcept(s);
}

void Dictionary::addConcept(const std::string& s){
  uint32_t h = hash(s, 0);
  int32_t num = find(s, h, 0, entry_type::kbconcept);

  if (word2int_[num] == -1) {
    entry e;
    e.word = s;
    e.pos_tag = 0;
    e.count = 1;
    e.type = entry_type::kbconcept;
    words_.push_back(e);
    word2int_[num] = size_++;
  } else {
    words_[word2int_[num]].count++;
  }
}

void Dictionary::addWord(const word_t &w){
  uint32_t h = hash(w.word_id, w.pos_tag);
  int32_t num = find(w.word_id, h, w.pos_tag, entry_type::word);

  ntokens_++;
  if (word2int_[num] == -1) {
    entry e;
    e.word = w.word_id;
    e.word_str = w.str;
    e.pos_tag = w.pos_tag;
    e.count = 1;
    // e.type = getType(w);
    e.type = entry_type::word;
    words_.push_back(e);
    word2int_[num] = size_++;
  } else {
    words_[word2int_[num]].count++;
  }
}
void Dictionary::addPhrase(const phrase_t& p, const sent_t::words_array_t& words){
  uint32_t h = hash(p.word_id);
  int32_t num = find(p.word_id, h, 0, entry_type::phrase);

  ntokens_++;
  if (word2int_[num] == -1) {
    entry e;
    e.word = p.word_id;
    e.pos_tag = 0;
    e.count = 1;
    e.type = entry_type::phrase;
    for(int i = 0;i<p.sz;++i) {
      const auto& w = words[p.components[i]];
      uint32_t wh = hash(w.word_id, w.pos_tag);
      e.hashes.push_back(wh);
      e.subwords.push_back(find(w.word_id, wh, w.pos_tag));
    }

    words_.push_back(e);
    word2int_[num] = size_++;
  } else {
    words_[word2int_[num]].count++;
  }
}

std::pair<uint32_t, int32_t> Dictionary::addSubword(const std::string& word) {
  uint32_t h = hash(word);
  int32_t pos = find(word, h, 0, entry_type::subword);
  if (word2int_[pos] == -1) {
    entry e;
    e.word = word;
    e.count = 1;
    e.pos_tag = 0;
    e.type = entry_type::subword;
    words_.push_back(e);
    word2int_[pos] = size_++;
    nsubwords_++;
  } else {
    words_[word2int_[pos]].count++;
  }
  return std::pair(h, pos);
}

int32_t Dictionary::size(entry_type types) const {
  int32_t sz = 0;
  if (contains(types, entry_type::word))
    sz += nwords_;
  if (contains(types, entry_type::phrase))
    sz += nphrases_;
  if (contains(types, entry_type::label))
    sz += nlabels_;
  if (contains(types, entry_type::kbconcept))
    sz += nkbconcepts_;
  if (contains(types, entry_type::subword))
    sz += nsubwords_;
  return sz;
}



int32_t Dictionary::nwords() const {
  return nwords_;
}

int32_t Dictionary::nlabels() const {
  return nlabels_;
}

int64_t Dictionary::ntokens() const {
  return ntokens_;
}

const std::vector<int32_t>& Dictionary::getSubwords(int32_t i) const {
  assert(i >= 0);
  assert(i < size_);
  return words_[i].subwords;
}

const std::vector<int32_t>
Dictionary::getSubwords(const std::string& word, uint8_t pos_tag) const {
  int32_t i = getId(word, pos_tag);
  if (i >= 0) {
    return getSubwords(i);
  }
  std::vector<int32_t> ngrams;
  if (word != EOS) {
    computeSubwords(word, ngrams);
  }
  return ngrams;
}

void Dictionary::getSubwords(
    const std::string& word,
    std::vector<int32_t>& ngrams,
    std::vector<std::string>& substrings) const {
  int32_t i = getId(word);
  ngrams.clear();
  substrings.clear();
  if (i >= 0) {
    ngrams.push_back(i);
    substrings.push_back(words_[i].word);
  }
  if (word != EOS) {
    computeSubwords(word, ngrams, &substrings);
  }
}

bool Dictionary::discard(int32_t id, real rand) const {
  assert(id >= 0);
  if (args_->model == model_name::sup) {
    return false;
  }
  return rand > pdiscard_[id];
}

int32_t Dictionary::getId(const std::string& w, uint32_t h) const {
  int32_t id = find(w, h);
  return word2int_[id];
}

int32_t Dictionary::getId(const std::string& w, uint8_t pos_tag, entry_type et) const {
  int32_t h = find(w, pos_tag, et);
  return word2int_[h];
}

entry_type Dictionary::getType(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].type;
}

entry_type Dictionary::getType(const std::string& w) const {
  return (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
}

std::string Dictionary::getWord(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].word;
}
int Dictionary::getPoS(uint32_t id) const{
  return words_[id].pos_tag;
}


// The correct implementation of fnv should be:
// h = h ^ uint32_t(uint8_t(str[i]));
// Unfortunately, earlier version of fasttext used
// h = h ^ uint32_t(str[i]);
// which is undefined behavior (as char can be signed or unsigned).
// Since all fasttext models that were already released were trained
// using signed char, we fixed the hash function to make models
// compatible whatever compiler is used.
uint32_t Dictionary::hash(const std::string_view& str) const {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(int8_t(str[i]));
    h = h * 16777619;
  }
  return h;
}
uint32_t Dictionary::hash(const std::string_view& str, uint8_t pos_tag) const {
  uint32_t h = hash(str);
  if(pos_tag > 0){
    h = h  ^ (pos_tag << 6);
    h = h * 16777619;
  }
  return h;
}

void Dictionary::computeSubwords(
    const std::string& word,
    std::vector<int32_t>& ngrams,
    std::vector<std::string>* substrings) const {
  auto subwords = extractSubwords(word);
  for(const auto& subword : subwords){
    auto id = find(subword, 0, entry_type::subword);
    if (word2int_[id] == -1)
      continue;
    ngrams.push_back(word2int_[id]);
  }

  if(substrings)
    *substrings = std::move(subwords);
}

std::vector<std::string>
Dictionary::extractSubwords(const std::string& s)const{
  auto variants = encoder_->apply(s, args_->maxBpeVars);
  return fastBPE::uniq_subwords(variants, args_->minn);
}

void Dictionary::initSubwords(){
  //this function is invoked when creating dict first time
  auto sz = size_;
  int64_t minThreshold = 1;
  for (size_t i = 0; i < sz; i++) {
    if(words_[i].type != entry_type::word)
      continue;
    words_[i].subwords.clear();
    words_[i].subwords.push_back(i);
    if (words_[i].word_str != EOS) {
      auto subwords = extractSubwords(words_[i].word_str);

      for(const auto& subword : subwords){
        auto [h, pos] = addSubword(subword);
        words_[i].subwords.push_back(pos);
        words_[i].hashes.push_back(h);
      }

      if (size_ > 0.75 * MAX_VOCAB_SIZE) {
        minThreshold++;
        threshold(minThreshold, minThreshold);
      }


    }
  }
}


void Dictionary::initNgrams() {
  for (size_t i = 0; i < size_; i++) {
    std::string word = BOW + words_[i].word + EOW;
    words_[i].subwords.clear();
    words_[i].subwords.push_back(i);
    if (words_[i].word != EOS) {
      computeSubwords(word, words_[i].subwords);
    }
  }
}

bool Dictionary::readWord(std::istream& in, std::string& word) const {
  int c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  while ((c = sb.sbumpc()) != EOF) {
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
        c == '\f' || c == '\0') {
      if (word.empty()) {
        if (c == '\n') {
          word += EOS;
          return true;
        }
        continue;
      } else {
        if (c == '\n')
          sb.sungetc();
        return true;
      }
    }
    word.push_back(c);
  }
  // trigger eofbit
  in.get();
  return !word.empty();
}



void Dictionary::addSent(const sent_t& sent){
  for (const auto& w : sent.words){
    addWord(w);
    if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
      std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::flush;
    }

  }
  for(const auto& p : sent.phrases){
    if (p.sz == 0)
      continue;
    addPhrase(p, sent.words);
    if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
      std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::flush;
    }

  }

  addConcepts(sent.concepts);
}

void Dictionary::addLine(const line_t& line){
  addSent(line.target);
  for (const auto& os : line.other_langs)
    addSent(os);
}

void Dictionary::readFromFile(std::istream& in) {
  int64_t minThreshold = 1;
  std::string json;
  line_t line;
  while (std::getline(in, json)){
    parse_from_json(json, line);

    addLine(line);
    if (size_ > 0.75 * MAX_VOCAB_SIZE) {
      minThreshold++;
      threshold(minThreshold, minThreshold);
    }
  }
  threshold(args_->minCount, args_->minCountLabel);
  std::cerr<<"# words "<<nwords_<<std::endl;

  initTableDiscard();
  std::cerr<<"\rinit subwords"<<std::endl;
  initSubwords();
  std::cerr<<"# subwords "<<nsubwords_<<std::endl;
  if (args_->verbose > 0) {
    std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::endl;
    std::cerr << "Number of words:  " << nwords_ << std::endl;
    std::cerr << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    throw std::invalid_argument(
        "Empty vocabulary. Try a smaller -minCount value.");
  }
}

void Dictionary::threshold(int64_t t, int64_t tl) {
  std::cerr<<"threshold dictionary: word_cnt="<<t<<" label cnt="<<tl<<std::endl;
  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
    if (e1.type != e2.type) {
      return e1.type < e2.type;
    }
    return e1.count > e2.count;
  });
  words_.erase(
      remove_if(
          words_.begin(),
          words_.end(),
          [&](const entry& e) {
            //TODO specific count for subwords and phrases?
            return (e.type != entry_type::label && e.count < t) ||
                (e.type == entry_type::label && e.count < tl);
          }),
      words_.end());
  words_.shrink_to_fit();
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  nsubwords_ = 0;
  nphrases_ = 0;
  nkbconcepts_ = 0;
  std::fill(word2int_.begin(), word2int_.end(), -1);
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word, it->pos_tag, it->type);
    word2int_[h] = size_++;
    if (it->type == entry_type::word) {
      //TODO init subwords
      nwords_++;
    }
    if (it->type == entry_type::subword) {
      nsubwords_++;
    }
    if (it->type == entry_type::phrase) {
      //TODO init subwords
      nphrases_++;
    }
    if (it->type == entry_type::label) {
      nlabels_++;
    }
    if (it->type == entry_type::kbconcept){
      nkbconcepts_++;
    }
  }
}

void Dictionary::initTableDiscard() {
  pdiscard_.resize(size_, 1.);
  auto words_or_phrases = combine(entry_type::word, entry_type::phrase);

  for (size_t i = 0; i < size_; i++) {
    if (contains(words_or_phrases, words_[i].type)){
      real f = real(words_[i].count) / real(ntokens_);
      pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
    }
  }
}

std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
  std::vector<int64_t> counts;
  for (auto& w : words_) {
    if (contains(type, w.type)) {
      counts.push_back(w.count);
    }
  }
  return counts;
}

void Dictionary::addWordNgrams(
    std::vector<int32_t>& line,
    const std::vector<int32_t>& hashes,
    int32_t n) const {
  for (int32_t i = 0; i < hashes.size(); i++) {
    uint64_t h = hashes[i];
    for (int32_t j = i + 1; j < hashes.size() && j < i + n; j++) {
      h = h * 116049371 + hashes[j];
      pushHash(line, h % args_->bucket);
    }
  }
}

void Dictionary::addSubwords(
    std::vector<int32_t>& line,
    const std::string& token,
    int32_t wid) const {
  if (wid < 0) { // out of vocab
    if (token != EOS) {
      computeSubwords(BOW + token + EOW, line);
    }
  } else {
    if (args_->maxn <= 0) { // in vocab w/o subwords
      line.push_back(wid);
    } else { // in vocab w/ subwords
      const std::vector<int32_t>& ngrams = getSubwords(wid);
      line.insert(line.end(), ngrams.cbegin(), ngrams.cend());
    }
  }
}

void Dictionary::reset(std::istream& in) const {
  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }
}

int32_t Dictionary::getLine(
    std::istream& in,
    std::vector<int32_t>& words,
    std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  while (readWord(in, token)) {
    int32_t h = find(token);
    int32_t wid = word2int_[h];
    if (wid < 0) {
      continue;
    }

    ntokens++;
    if (getType(wid) == entry_type::word && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    if (ntokens > MAX_LINE_SIZE || token == EOS) {
      break;
    }
  }
  return ntokens;
}

int32_t Dictionary::getLine(std::istream& in,
                            compact_line_t& line,
                            std::minstd_rand& rng) const{
  //this function is used when training via skipgramp/syntax_skipgram

  std::uniform_real_distribution<> uniform(0, 1);
  int32_t ntokens = 0;

  std::string json;
  std::getline(in, json);
  if(in.eof()){
    reset(in);
    std::getline(in, json);
  }


  auto get_id_func = [this](const char* word_id, uint8_t pos_tag){
    auto h = find(word_id, pos_tag);
    return word2int_[h];
  };
  parse_from_json(json, get_id_func, line);
  make_aux_offs(line);
  if (not line.other_langs.empty() and
      line.other_langs.front().mapping_to_target_words.empty())
    fill_other_mapping_randomly(line);

  auto fin_sent = [this, &ntokens, &uniform, &rng](compact_sent_t& s){
    for(auto& w : s.words)
      if (w.num >= 0){
        ntokens++;
        if (discard(w.num, uniform(rng)))
          w.num = -1;
      }
    for(auto& w : s.phrases)
      if (w.is_phrase and w.num >= 0){
        ntokens++;
        if (discard(w.num, uniform(rng)))
          w.num = -1;
      }
  };

  fin_sent(line.target);
  for(auto& os : line.other_langs)
    fin_sent(os);
  return ntokens;

}

int32_t Dictionary::getLine(
    std::istream& in,
    std::vector<int32_t>& words,
    std::vector<int32_t>& labels) const {
  std::vector<int32_t> word_hashes;
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  labels.clear();
  while (readWord(in, token)) {
    uint32_t h = hash(token);
    int32_t wid = getId(token, h);
    entry_type type = wid < 0 ? getType(token) : getType(wid);

    ntokens++;
    if (type == entry_type::word) {
      addSubwords(words, token, wid);
      word_hashes.push_back(h);
    } else if (type == entry_type::label && wid >= 0) {
      labels.push_back(wid - nwords_);
    }
    if (token == EOS) {
      break;
    }
  }
  addWordNgrams(words, word_hashes, args_->wordNgrams);
  return ntokens;
}

void Dictionary::pushHash(std::vector<int32_t>& hashes, int32_t id) const {
  if (pruneidx_size_ == 0 || id < 0) {
    return;
  }
  if (pruneidx_size_ > 0) {
    if (pruneidx_.count(id)) {
      id = pruneidx_.at(id);
    } else {
      return;
    }
  }
  hashes.push_back(nwords_ + id);
}

std::string Dictionary::getLabel(int32_t lid) const {
  if (lid < 0 || lid >= nlabels_) {
    throw std::invalid_argument(
        "Label id is out of range [0, " + std::to_string(nlabels_) + "]");
  }
  return words_[lid + nwords_].word;
}

void Dictionary::save(std::ostream& out) const {
  out.write((char*)&size_, sizeof(int32_t));
  out.write((char*)&nwords_, sizeof(int32_t));
  out.write((char*)&nlabels_, sizeof(int32_t));
  out.write((char*)&nsubwords_, sizeof(int32_t));
  out.write((char*)&nphrases_, sizeof(int32_t));
  out.write((char*)&nkbconcepts_, sizeof(int32_t));
  out.write((char*)&ntokens_, sizeof(int64_t));
  out.write((char*)&pruneidx_size_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    entry e = words_[i];
    out.write(e.word.data(), e.word.size() * sizeof(char));
    out.put(0);
    out.put(e.pos_tag);
    uint16_t hsz = e.hashes.size();
    out.write((char*)&(hsz), 2);
    out.write((char*)e.hashes.data(),
              e.hashes.size() * sizeof(decltype(hash(""))));
    out.write((char*)&(e.count), sizeof(int64_t));
    out.write((char*)&(e.type), sizeof(entry_type));
  }
  for (const auto pair : pruneidx_) {
    out.write((char*)&(pair.first), sizeof(int32_t));
    out.write((char*)&(pair.second), sizeof(int32_t));
  }
  encoder_->save(out);

}

void Dictionary::load(std::istream& in) {
  words_.clear();
  in.read((char*)&size_, sizeof(int32_t));
  in.read((char*)&nwords_, sizeof(int32_t));
  in.read((char*)&nlabels_, sizeof(int32_t));
  in.read((char*)&nsubwords_, sizeof(int32_t));
  in.read((char*)&nphrases_, sizeof(int32_t));
  in.read((char*)&nkbconcepts_, sizeof(int32_t));
  in.read((char*)&ntokens_, sizeof(int64_t));
  in.read((char*)&pruneidx_size_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    char c;
    entry e;
    while ((c = in.get()) != 0) {
      e.word.push_back(c);
    }
    //word_str is not saved, not restored
    char p;
    in.get(p);
    e.pos_tag = p;
    uint16_t hsz;
    in.read((char*)&hsz, sizeof(uint16_t));
    uint32_t h;
    int k = 0;
    while(k++ < hsz){
      in.read((char*)&h, sizeof(h));
      e.hashes.push_back(h);
    }
    in.read((char*)&e.count, sizeof(int64_t));
    in.read((char*)&e.type, sizeof(entry_type));
    words_.push_back(e);
  }
  pruneidx_.clear();
  for (int32_t i = 0; i < pruneidx_size_; i++) {
    int32_t first;
    int32_t second;
    in.read((char*)&first, sizeof(int32_t));
    in.read((char*)&second, sizeof(int32_t));
    pruneidx_[first] = second;
  }
  encoder_->load(in);

  initTableDiscard();

  word2int_.assign(MAX_VOCAB_SIZE, -1);
  for (int32_t i = 0; i < size_; i++) {
    word2int_[find(words_[i].word, words_[i].pos_tag, words_[i].type)] = i;
  }
  initSubwordsPos();

  std::cerr<<"Loaded dict nwords="<<nwords_<<" phrases="<<nphrases_<<" concepts="<<nkbconcepts_
           <<" subwords="<<nsubwords_<<" ntokens="<<ntokens_<<std::endl;
}


void Dictionary::init() {
  //this method is used only in FastText::getInputMatrixFromFile
  initTableDiscard();
  // initNgrams();
}

void Dictionary::initSubwordsPos(){
  //this function is invoked every time dict is loaded
  for (size_t i = 0; i < size_; i++) {
    auto& w = words_[i];
    // if (w.type != entry_type::word and w.type != entry_type::phrase)
    if (w.type == entry_type::label)
      continue;

    auto find_type = entry_type::subword;
    if (w.type == entry_type::phrase)
      find_type = entry_type::word;

    w.subwords.clear();
    w.subwords.push_back(i);
    for(auto h : w.hashes){
      auto id = find(h, find_type);
      auto pos = word2int_[id];
      if (pos != -1)
        w.subwords.push_back(pos);
    }
  }
}

void Dictionary::prune(std::vector<int32_t>& idx) {
  std::vector<int32_t> words, ngrams;
  for (auto it = idx.cbegin(); it != idx.cend(); ++it) {
    if (*it < nwords_) {
      words.push_back(*it);
    } else {
      ngrams.push_back(*it);
    }
  }
  std::sort(words.begin(), words.end());
  idx = words;

  if (ngrams.size() != 0) {
    int32_t j = 0;
    for (const auto ngram : ngrams) {
      pruneidx_[ngram - nwords_] = j;
      j++;
    }
    idx.insert(idx.end(), ngrams.begin(), ngrams.end());
  }
  pruneidx_size_ = pruneidx_.size();

  std::fill(word2int_.begin(), word2int_.end(), -1);

  int32_t j = 0;
  for (int32_t i = 0; i < words_.size(); i++) {
    if (getType(i) == entry_type::label ||
        (j < words.size() && words[j] == i)) {
      words_[j] = words_[i];
      const auto& wt = words_[j];
      word2int_[find(wt.word, wt.pos_tag, wt.type)] = j;
      j++;
    }
  }
  nwords_ = words.size();
  size_ = nwords_ + nlabels_;
  words_.erase(words_.begin() + size_, words_.end());
  initNgrams();
}

void Dictionary::dump(std::ostream& out) const {
  out << words_.size() << std::endl;
  for (auto it = words_.cbegin(); it != words_.cend(); ++it) {
    std::string entryType;
    switch(it->type){
    case entry_type::word : entryType = "word"; break;
    case entry_type::label : entryType = "label"; break;
    case entry_type::subword : entryType = "subword"; break;
    case entry_type::phrase : entryType = "phrase"; break;
    case entry_type::kbconcept: entryType = "concept"; break;
    case entry_type::all: break;
    }
    auto num = it - words_.cbegin();
    auto h = hash(it->word, it->pos_tag);
    auto to_str = boost::adaptors::transformed([](auto i){return std::to_string(i);});
    out <<"# "<<num<<" "<<entryType<<": " << it->word <<" postag="<<(int)it->pos_tag
        << " h="<<h
        << " cnt=" << it->count
        << " sub_hashes=" << boost::join(it->hashes | to_str, ",")
        << " sub_nums=" << boost::join(it->subwords | to_str, ",")<< std::endl;
  }
}

} // namespace fasttext
