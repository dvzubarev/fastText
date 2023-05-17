#ifndef FASTTEXT_SENT_HXX
#define FASTTEXT_SENT_HXX

#include "sent.h"
#include <stdexcept>


#include <rapidjson/reader.h>
#include <rapidjson/error/en.h>

namespace fasttext {


inline bool mem_equal(const char* s1, size_t l1, const char* s2, size_t l2){
  return l1 != l2 ? false : std::memcmp(s1, s2, l1) == 0;
}

struct handler_traits_t{
  using line_type = line_t;
  using sent_type = sent_t;
  using word_type = word_t;
  using phrase_type = phrase_t;
};

using utf_traits_t = rapidjson::UTF8<>;
template<class Impl, class Traits>
struct base_line_handler_t : public rapidjson::BaseReaderHandler<utf_traits_t,
                                                                 base_line_handler_t<Impl, Traits>> {
  static constexpr size_t SENT_SIZE_HINT = 50;
  static constexpr size_t CONCEPTS_SIZE_HINT = 5;
  using sz_t = rapidjson::SizeType;
  enum state : int{
    sLineObjBegin,
    sLineObj,
    sSentObj,
    sOtherSentObjBegin,
    sOtherSentObj,
    sWordsArrBegin,
    sWordObjBegin,
    sPhrasesArrBegin,
    sPhraseObjBegin,
    sConceptsArr,
    sComponent,
    sWordObj,
    sWordStrVal,
    sWordIdVal,
    sWordPosTagVal,
    sWordParentOffsVal,
    sWordSyntRelVal,
    sOtherSentArr,
    sWordsMappingArr,
    sPhrasesMappingArr,
    sOrigin,
  };
  union val_t{
    bool b;
    int i;
    unsigned u;
    int64_t i64;
    uint64_t u64;
    double d;
  };

  using line_type = typename Traits::line_type;
  using sent_type = typename Traits::sent_type;
  using word_type = typename Traits::word_type;
  using phrase_type = typename Traits::phrase_type;


  line_type* line_;
  sent_type* sent_;
  word_type* word_;
  phrase_type* phrase_;

  state state_;
  bool target_sent_;


  const char* key_;
  sz_t sz_;
  const char* val_;
  val_t num_val_;

  std::string err_;


  base_line_handler_t(line_type& line):state_(sLineObjBegin), line_(&line) {}

  Impl& _impl(){
    return static_cast<Impl&>(*this);
  }

  inline word_type* current_word(){
    return word_ ? word_ : phrase_;
  }

  inline bool on_line_obj_begin(){
    line_->reset();
    state_ = sLineObj;
    return true;
  }
  bool on_line_obj(){
    if(mem_equal("target", 6u, key_, sz_)){
      target_sent_ = true;
      state_ = sSentObj;
      sent_ = &line_->target;
      return true;
    }
    if(mem_equal("other_langs", 11u, key_, sz_)){
      target_sent_ = false;
      state_ = sOtherSentArr;
      return true;
    }

    err_ = "Unknown line obj key: ";
    err_.append(key_, sz_);
    return false;
  }

  inline bool on_sent_obj(){
    if(mem_equal("words", 5u, key_, sz_)){
      state_ = sWordsArrBegin;
      return true;
    }
    if (mem_equal("phrases", 7u, key_, sz_)){
      state_ = sPhrasesArrBegin;
      return true;
    }
    if (mem_equal("concepts", 8u, key_, sz_)){
      state_ = sConceptsArr;
      return true;
    }
    if (mem_equal("origin", 6u, key_, sz_)){
      state_ = sOrigin;
      return true;
    }

    err_ = "Unknown sent obj key: ";
    err_.append(key_, sz_);
    return false;
  }

  inline bool on_other_sent_obj(){
    // return on_sent_obj();
    if(on_sent_obj())
      return true;

    err_ = "";

    if (mem_equal("words_mapping", 13u, key_, sz_)){
      state_ = sWordsMappingArr;
      return true;
    }

    if (mem_equal("phrases_mapping", 15u, key_, sz_)){
      state_ = sPhrasesMappingArr;
      return true;
    }

    err_ = "Unknown other sent obj key: ";
    err_.append(key_, sz_);
    return false;
  }

  inline bool on_words_arr_begin(){
    sent_->words.reserve(SENT_SIZE_HINT);
    state_ = sWordObjBegin;
    return true;
  }
  inline bool on_word_obj_begin(){
    sent_->words.emplace_back();
    word_ = &sent_->words.back();
    state_ = sWordObj;
    return true;
  }
  inline bool on_phrase_arr_begin(){
    sent_->phrases.reserve(SENT_SIZE_HINT);
    state_ = sPhraseObjBegin;
    return true;
  }
  inline bool on_phrase_obj_begin(){
    sent_->phrases.emplace_back();
    phrase_ = &sent_->phrases.back();
    state_ = sWordObj;
    return true;
  }
  bool on_word_obj(){
    assert(sz_ == 1);
    switch(static_cast<char>(*key_)){
    case 'w':
      state_ = sWordStrVal;
      return true;
    case 'i':
      state_ = sWordIdVal;
      return true;
    case 'p':
      state_ = sWordPosTagVal;
      return true;
    case 'l':
      state_ = sWordParentOffsVal;
      return true;
    case 'n':
      state_ = sWordSyntRelVal;
      return true;
    case 'C':
      state_ = sComponent;
      return true;
    }

    err_ = "Unknown word obj key: ";
    err_.append(key_, sz_);
    return false;
  }
  inline bool on_word_str_val(){
    auto ret = _impl().set_word_str();
    state_ = sWordObj;
    return ret;
  }
  inline bool on_word_id_val(){
    auto ret = _impl().set_word_id();
    state_ = sWordObj;
    return ret;
  }
  inline bool on_word_pos_tag_val(){
    auto ret = _impl().set_pos_tag();
    state_ = sWordObj;
    return ret;
  }
  inline bool on_word_parent_offs_val(){
    auto ret = _impl().set_parent_offs();
    state_ = sWordObj;
    return ret;
  }
  inline bool on_word_synt_rel_val(){
    auto ret = _impl().set_synt_rel();
    state_ = sWordObj;
    return ret;
  }
  inline bool on_components_begin(){
    return _impl().components_begin();
  }
  inline bool on_component_val(){
    return _impl().set_component();
  }
  inline bool on_concepts_begin(){
    return _impl().concepts_begin();
  }
  inline bool on_concept_val(){
    return _impl().set_concept();
  }

  inline bool on_words_mapping_begin(){
    return _impl().words_mapping_begin();
  }
  inline bool on_words_mapping_val(){
    return _impl().set_words_mapping();
  }
  inline bool on_phrases_mapping_begin(){
    return _impl().phrases_mapping_begin();
  }
  inline bool on_phrases_mapping_val(){
    return _impl().set_phrases_mapping();
  }

  inline bool on_other_sent_obj_begin(){
    line_->other_langs.emplace_back();
    sent_ = &line_->other_langs.back();
    state_ = sOtherSentObj;
    return true;
  }

  bool handle_state(){
    switch(state_){
    case sLineObj: return on_line_obj();
    case sSentObj: return on_sent_obj();
    case sOtherSentObj: return on_other_sent_obj();
    case sConceptsArr: return on_concept_val();
    case sComponent: return on_component_val();
    case sWordObj: return on_word_obj();
    case sWordStrVal: return on_word_str_val();
    case sWordIdVal: return on_word_id_val();
    case sWordPosTagVal: return on_word_pos_tag_val();
    case sWordParentOffsVal: return on_word_parent_offs_val();
    case sWordSyntRelVal: return on_word_synt_rel_val();
    case sWordsMappingArr: return on_words_mapping_val();
    case sPhrasesMappingArr: return on_phrases_mapping_val();
    case sOrigin: state_ = target_sent_ ? sSentObj : sOtherSentObj; return true;
    default: {
      err_ = "Unknown state in handle state " + std::to_string(state_) ;
      return false;
    }
    }

  }

  bool Null() { err_ = "Null encountered!"; return false; }
  bool Bool(bool b) {
    num_val_.b = b;
    return handle_state();
  }
  bool Int(int i) {
    num_val_.i = i;
    return handle_state();
  }
  bool Uint(unsigned u) {
    num_val_.u = u;
    return handle_state();
  }
  bool Int64(int64_t i) {
    num_val_.i64 = i;
    return handle_state();
  }
  bool Uint64(uint64_t u) {
    num_val_.u64 = u;
    return handle_state();
  }
  bool Double(double d) {
    num_val_.d = d;
    return handle_state();
  }
  bool String(const char* str, sz_t length, bool copy) {
    val_ = str;
    sz_ = length;
    auto ret = handle_state();
    val_ = nullptr;
    sz_ = 0;
    return ret;
  }
  bool StartObject() {
    switch(state_){
    case sLineObjBegin: return on_line_obj_begin();
    case sSentObj: return true;
    case sOtherSentObjBegin: return on_other_sent_obj_begin();
    case sOtherSentObj: return true;
    case sWordObjBegin: return on_word_obj_begin();
    case sPhraseObjBegin: return on_phrase_obj_begin();
    default: {
      err_ = "Unknown state in startObject " + std::to_string(state_);
      return false;
    }
    }
  }
  bool Key(const char* str, sz_t length, bool copy) {
    key_ = str;
    sz_ = length;
    auto ret = handle_state();
    key_ = nullptr;
    sz_ = 0;
    return ret;
  }
  bool EndObject(sz_t memberCount) {

    switch(state_){
    case sLineObj: return true;
    case sSentObj:
    case sOtherSentObj: {
      state_ = target_sent_ ? sLineObj : sOtherSentObjBegin;
      return true;

    }
    case sWordObj: {
      auto ret = _impl().word_object_end();
      state_ = word_ ? sWordObjBegin : sPhraseObjBegin;
      word_ = nullptr;
      phrase_ = nullptr;
      return ret;
    }
    default: {
      err_ = "Unknown state in EndObject " + std::to_string(state_) ;
      return false;
    }
    }
  }
  bool StartArray() {
    switch(state_){
    case sWordsArrBegin: return on_words_arr_begin();
    case sPhrasesArrBegin: return on_phrase_arr_begin();
    case sConceptsArr: return on_concepts_begin();
    case sComponent: return on_components_begin();
    case sWordsMappingArr: return on_words_mapping_begin();
    case sPhrasesMappingArr: return on_phrases_mapping_begin();
    case sOtherSentArr: state_ = sOtherSentObjBegin; return true;
    default : {
      err_ = "Unknown state in Start array " + std::to_string(state_) ;
      return false;

    }
    }
  }
  bool EndArray(sz_t elementCount) {
    switch(state_){
    case sOtherSentObjBegin: state_ = sLineObj; return true;
    case sWordObjBegin:
    case sPhraseObjBegin:
    case sConceptsArr: state_ = target_sent_ ? sSentObj : sOtherSentObj; return true;
    case sComponent: state_ = sWordObj; return true;
    case sWordsMappingArr: state_ = sOtherSentObj; return true;
    case sPhrasesMappingArr: state_ = sOtherSentObj; return true;
    default: {
      err_ = "Unknown state in End array " + std::to_string(state_) ;
      return false;
    }
    }

  }
};

struct line_handler_t : public base_line_handler_t<line_handler_t, handler_traits_t> {
  using base_t = base_line_handler_t<line_handler_t, handler_traits_t>;

  using base_t::base_t;

  inline bool set_word_str(){
    word_t* w = current_word();
    w->str = val_;
    return true;
  }
  inline bool set_word_id(){
    word_t* w = current_word();
    w->word_id = val_;
    return true;
  }
  inline bool set_pos_tag(){
    word_t* w = current_word();
    w->pos_tag = num_val_.u;
    return true;
  }
  inline bool set_parent_offs(){
    word_t* w = current_word();
    w->parent_offs = num_val_.i;
    return true;
  }
  inline bool set_synt_rel(){
    word_t* w = current_word();
    w->synt_rel = num_val_.u;
    return true;
  }

  inline bool components_begin(){
    auto& cmps = phrase_->components;
    std::fill( std::begin( cmps ), std::end( cmps ), -1 );
    phrase_->sz = 0;
    return true;
  }
  inline bool set_component(){
    phrase_->components[phrase_->sz++] = num_val_.i;
    return true;
  }
  inline bool concepts_begin(){
    sent_->concepts.reserve(CONCEPTS_SIZE_HINT);
    return true;
  }
  inline bool set_concept(){
    sent_->concepts.push_back(val_);
    return true;
  }

  inline bool words_mapping_begin(){
    return true;
  }
  inline bool set_words_mapping(){
    return true;
  }
  inline bool phrases_mapping_begin(){
    return true;
  }
  inline bool set_phrases_mapping(){
    return true;
  }


  inline bool word_object_end(){
    word_t* w = current_word();
    if (w->word_id == nullptr)
      w->word_id = w->str;
    return true;
  }

};



struct compact_handler_traits_t{
  using line_type = compact_line_t;
  using sent_type = compact_sent_t;
  using word_type = compact_word_t;
  using phrase_type = compact_word_t;
};

template <class GetIdFunc>
struct compact_line_handler_t : public base_line_handler_t<compact_line_handler_t<GetIdFunc>,
                                                           compact_handler_traits_t> {
  using base_t = base_line_handler_t<compact_line_handler_t<GetIdFunc>, compact_handler_traits_t>;
  using line_type = typename base_t::line_type;
  using word_type = typename base_t::word_type;

  compact_line_handler_t(const GetIdFunc& cb, line_type& line):base_t(line),
                                                               get_id_func_(cb){}

  const GetIdFunc& get_id_func_;
  const char* cur_id_ = nullptr;
  const char* cur_str_ = nullptr;
  uint8_t cur_pos_tag_ = 0;

  other_compact_sent_t* current_other_sent(){
    return static_cast<other_compact_sent_t*>(this->sent_);
  }

  inline bool set_word_str(){
    cur_str_ = this->val_;
    return true;
  }
  inline bool set_word_id(){
    cur_id_ = this->val_;
    return true;
  }
  inline bool set_pos_tag(){
    cur_pos_tag_ = this->num_val_.u;
    return true;
  }
  inline bool set_parent_offs(){
    word_type* w = this->current_word();
    w->parent_offs = compact_word_t::parent_offs_to_bits(this->num_val_.i);
    return true;
  }
  inline bool set_synt_rel(){
    word_type* w = this->current_word();
    if (this->num_val_.u >= 32)
      w->synt_rel = 31;
    else
      w->synt_rel = this->num_val_.u;
    return true;
  }

  inline bool components_begin(){
    this->phrase_->is_phrase = true;
    return true;
  }
  inline bool set_component(){
    return true;
  }

  inline bool concepts_begin(){
    this->sent_->concepts.reserve(base_t::CONCEPTS_SIZE_HINT);
    return true;
  }
  inline bool set_concept(){
    int32_t num = get_id_func_(this->val_, 0);
    if (num != -1)
      this->sent_->concepts.push_back(num);
    return true;
  }

  inline bool words_mapping_begin(){
    auto sent = current_other_sent();
    if (not sent->words.empty())
      sent->mapping_to_target_words.reserve(sent->words.size());
    return true;
  }
  inline bool set_words_mapping(){
    auto sent = current_other_sent();
    sent->mapping_to_target_words.push_back(this->num_val_.i);
    return true;
  }
  inline bool phrases_mapping_begin(){
    auto sent = current_other_sent();
    if (not sent->phrases.empty())
      sent->mapping_to_target_phrases.reserve(sent->phrases.size());
    return true;
  }
  inline bool set_phrases_mapping(){
    auto sent = current_other_sent();
    sent->mapping_to_target_phrases.push_back(this->num_val_.i);
    return true;
  }

  inline bool word_object_end(){
    if (!cur_id_ && !cur_str_){
      this->err_ = "word_id and word str is empty!";
      return false;
    }

    int32_t num = get_id_func_((cur_id_ == nullptr) ? cur_str_ : cur_id_, cur_pos_tag_);
    this->current_word()->num = num;

    cur_id_ = nullptr;
    cur_str_ = nullptr;
    cur_pos_tag_ = 0;

    return true;
  }

};

template<class Handler>
void parse(std::string& json, Handler& handler){

  rapidjson::Reader reader;
  rapidjson::InsituStringStream ss(json.data());

  auto ret = reader.Parse<rapidjson::kParseInsituFlag>(ss, handler);
  if(ret.IsError()){
    if(ret.Code() == rapidjson::kParseErrorTermination)
      throw std::runtime_error("Data error: " + handler.err_);
    else
      throw std::runtime_error("err code = " + std::to_string(ret.Code()) + \
                               ": " + rapidjson::GetParseError_En(ret.Code()));
  }

}

void parse_from_json(std::string& json, line_t& line){
  line_handler_t handler(line);
  parse(json, handler);
}


template<class GetIdFunc>
void parse_from_json(std::string& json,
                     const GetIdFunc& cb,
                     compact_line_t& line){

  compact_line_handler_t<GetIdFunc> handler(cb, line);
  parse(json, handler);
}


}


#endif // FASTTEXT_SENT_HXX
