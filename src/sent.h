#ifndef FASTTEXT_SENT_HPP
#define FASTTEXT_SENT_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <array>
#include <algorithm>

namespace fasttext {

struct word_t{
  uint8_t pos_tag = 0;
  uint8_t synt_rel = 0;
  int16_t parent_offs;

  // uint64_t hash_;
  const char* str = nullptr;
  const char* word_id = nullptr;

};

struct compact_word_t{
  constexpr static int BITS_PER_PARENT_OFFS = 8;
  constexpr static int BITS_PER_OFFS = 6;

  static inline int8_t offs_to_bits_impl(int i, const int bits_per_offs){
    if (std::abs(i) <= (1<<(bits_per_offs-1))-1)
      return i;
    return 0;
  }
  static inline int8_t offs_to_bits(int i){
    return offs_to_bits_impl(i, BITS_PER_OFFS);
  }
  static inline int8_t parent_offs_to_bits(int i){
    return offs_to_bits_impl(i, BITS_PER_PARENT_OFFS);
  }

  uint32_t is_phrase:1 = false;
  uint32_t synt_rel:5 = 0;
  int32_t parent_offs:8 = 0;
  int32_t first_child_offs:6 = 0;
  int32_t prev_sibling_offs:6 = 0;
  int32_t next_sibling_offs:6 = 0;

  int32_t num;
};

void make_aux_offs(std::vector<compact_word_t>& words);


struct phrase_t : public word_t{
  static constexpr size_t MAX_PHRASE_SIZE = 10;

  std::array<int16_t, MAX_PHRASE_SIZE> components;
  uint8_t sz = 0;

};

struct sent_t{
  using words_array_t = std::vector<word_t>;
  using phrases_array_t = std::vector<phrase_t>;


  void reset();

  uint8_t lang;

  words_array_t words;
  phrases_array_t phrases;

  std::vector<std::string> concepts;

};

struct compact_sent_t{
  using words_array_t = std::vector<compact_word_t>;

  void reset();
  words_array_t words;
  words_array_t phrases;

  std::vector<int32_t> concepts;
};

struct other_compact_sent_t : public compact_sent_t{
  std::vector<int16_t>  mapping_to_target_words;
  std::vector<int16_t>  mapping_to_target_phrases;

  void reset();
};

struct line_t{
  void reset();

  sent_t target;
  std::vector<sent_t> other_langs;
};

struct compact_line_t{
  void reset();

  compact_sent_t target;
  std::vector<other_compact_sent_t> other_langs;

};

void make_aux_offs(compact_line_t& line);
void fill_other_mapping_randomly(compact_line_t& line);


bool contains(const std::vector<compact_word_t>& words, int32_t num);


void parse_from_json(std::string& json, line_t& line);
template<class GetIdFunc>
void parse_from_json(std::string& json,
                     const GetIdFunc& cb,
                     compact_line_t& line);

}




#endif // FASTTEXT_SENT_HPP
