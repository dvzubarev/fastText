#ifndef FASTTEXT_SENT_HXX
#define FASTTEXT_SENT_HXX

#include "sent.h"

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
  using sz_t = rapidjson::SizeType;
  enum state : int{
    sLineObjBegin,
    sLineObj,
    sSentObj,
    sWordsArrBegin,
    sWordObjBegin,
    sPhrasesArrBegin,
    sPhraseObjBegin,
    sComponent,
    sWordObj,
    sWordStrVal,
    sWordPosTagVal,
    sWordParentOffsVal,
    sWordSyntRelVal,
    sSentArr,
    sSentObjBegin
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
      state_ = sSentArr;
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
    err_ = "Unknown sent obj key: ";
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
    //TODO if make keys uint8, i can create table for selecting state.
    if (mem_equal("word", 4u, key_, sz_)){
      state_ = sWordStrVal;
      return true;
    }
    if (mem_equal("p", 1u, key_, sz_)){
      state_ = sWordPosTagVal;
      return true;
    }
    if (mem_equal("l", 1u, key_, sz_)){
      state_ = sWordParentOffsVal;
      return true;
    }
    if (mem_equal("n", 1u, key_, sz_)){
      state_ = sWordSyntRelVal;
      return true;
    }
    if (mem_equal("CO", 2u, key_, sz_)){
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

  inline bool on_sent_obj_begin(){
    line_->other_langs.emplace_back();
    sent_ = &line_->other_langs.back();
    state_ = sSentObj;
    return true;
  }

  bool handle_state(){
    switch(state_){
    case sLineObj: return on_line_obj();
    case sSentObj: return on_sent_obj();
    case sWordObj: return on_word_obj();
    case sWordStrVal: return on_word_str_val();
    case sWordPosTagVal: return on_word_pos_tag_val();
    case sWordParentOffsVal: return on_word_parent_offs_val();
    case sWordSyntRelVal: return on_word_synt_rel_val();
    case sComponent: return on_component_val();
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
    //skip target obj opening
    case sSentObj: return true;
    case sWordObjBegin: return on_word_obj_begin();
    case sPhraseObjBegin: return on_phrase_obj_begin();
    case sSentObjBegin: return on_sent_obj_begin();
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
    case sSentObj: {
      state_ = target_sent_ ? sLineObj : sSentObjBegin;
      return true;

    }
    case sWordObj: {
      auto ret = _impl().word_object_end();
      state_ = word_ ? sWordObjBegin : sPhraseObjBegin;
      word_ = nullptr;
      phrase_ = nullptr;
      return ret;
    }
    case sLineObj: return true;
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
    case sComponent: return on_components_begin();
    case sSentArr: state_ = sSentObjBegin; return true;
    default : {
      err_ = "Unknown state in Start array " + std::to_string(state_) ;
      return false;

    }
    }
  }
  bool EndArray(sz_t elementCount) {
    switch(state_){
    case sWordObjBegin:
    case sPhraseObjBegin: state_ = sSentObj; return true;
    case sComponent: state_ = sWordObj; return true;
    case sSentObjBegin: state_ = sLineObj; return true;
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

  inline bool word_object_end(){
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
  const char* cur_str_ = nullptr;
  uint8_t cur_pos_tag_ = 0;


  inline bool set_word_str(){
    cur_str_ = this->val_;
    return true;
  }
  inline bool set_pos_tag(){
    cur_pos_tag_ = this->num_val_.u;
    return true;
  }
  inline bool set_parent_offs(){
    word_type* w = this->current_word();
    w->parent_offs = compact_word_t::offs_to_bits(this->num_val_.i);
    return true;
  }
  inline bool set_synt_rel(){
    word_type* w = this->current_word();
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

  inline bool word_object_end(){
    if (!cur_str_ or !cur_pos_tag_){
      this->err_ = "str or pos_tag is empty!";
      return false;
    }

    int32_t num = get_id_func_(cur_str_, cur_pos_tag_);
    this->current_word()->num = num;

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
