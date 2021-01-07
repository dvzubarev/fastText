#ifndef FASTTEXT_SENT_HPP
#define FASTTEXT_SENT_HPP

#include <vector>
#include <cstdint>
#include <array>
#include <algorithm>

namespace fasttext {

struct word_t{
  uint8_t pos_tag;
  uint8_t synt_rel;
  int16_t parent_offs;

  // uint64_t hash_;
  const char* str;

};

struct compact_word_t{
  constexpr static int BITS_PER_OFFS = 6;
  static inline int8_t offs_to_bits(int i){
    if (std::abs(i) <= (1<<(BITS_PER_OFFS-1))-1)
      return i;
    return 0;
  }

  uint32_t is_phrase:1 = false;
  int32_t _reserved1:1;
  uint32_t synt_rel:6 = 0;
  int32_t parent_offs:6 = 0;
  int32_t first_child_offs:6 = 0;
  int32_t prev_sibling_offs:6 = 0;
  int32_t next_sibling_offs:6 = 0;

  int32_t num;
};

void make_aux_offs(std::vector<compact_word_t>& words);


struct phrase_t : public word_t{
  static constexpr size_t MAX_PHRASE_SIZE = 3;

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
