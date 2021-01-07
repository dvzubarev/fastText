#include "sent.h"

#include <numeric>


namespace fasttext {


void sent_t::reset(){
  words.clear();
  phrases.clear();
  concepts.clear();
}

void line_t::reset(){
  target.reset();
  other_langs.clear();
}

void compact_sent_t::reset(){
  words.clear();
  phrases.clear();
  concepts.clear();
}

void other_compact_sent_t::reset(){
  compact_sent_t::reset();
  mapping_to_target_words.clear();
  mapping_to_target_phrases.clear();
}

void compact_line_t::reset(){
  target.reset();
  other_langs.clear();
}

bool contains(const std::vector<compact_word_t>& words, int32_t num){
  auto pos = std::find_if(std::begin(words), std::end(words),
                          [num](const compact_word_t& w) {return w.num == num;});
  return (pos != std::end(words));
}

int32_t compute_offs(int32_t head_pos, int32_t dep_pos){
  auto diff = head_pos - dep_pos;
  return compact_word_t::offs_to_bits(diff);
}

void make_aux_offs(std::vector<compact_word_t>& words){
  for(int i = 0; i < words.size(); ++i){
    auto& head = words[i];
    int prev_sibling_pos = -1;

    for(int j = 0; j < words.size(); ++j){
      auto& mod = words[j];
      if (i == j or j + mod.parent_offs != i)
        continue;
      if (not head.first_child_offs)
        head.first_child_offs = compute_offs(j, i);
      if(prev_sibling_pos != -1){
        mod.prev_sibling_offs = compute_offs(prev_sibling_pos, j);
        words[prev_sibling_pos].next_sibling_offs = compute_offs(j, prev_sibling_pos);
      }
      prev_sibling_pos = j;
    }
  }
}

void make_aux_offs(compact_line_t& line){
  make_aux_offs(line.target.words);
  make_aux_offs(line.target.phrases);

  for(auto& s : line.other_langs){
    make_aux_offs(s.words);
    make_aux_offs(s.phrases);
  }
}

void fill_other_mapping_impl(const std::vector<compact_word_t>& target,
                             const std::vector<compact_word_t>& other,
                             std::vector<int16_t>& mapping){
  mapping.resize(other.size(), -1);
  auto sz = std::min(other.size(), target.size());
  std::iota(mapping.begin(), mapping.begin() + sz, 0);
  std::random_shuffle(mapping.begin(), mapping.begin() + sz);
}

void fill_other_mapping_randomly(compact_line_t& line){

  for(auto& s : line.other_langs){
    fill_other_mapping_impl(line.target.words, s.words, s.mapping_to_target_words);
    fill_other_mapping_impl(line.target.phrases, s.phrases, s.mapping_to_target_phrases);
  }
}


}
