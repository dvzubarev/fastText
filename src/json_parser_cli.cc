#include "sent.h"
#include "sent.hxx"

#include <iostream>
#include <fstream>

void print_sent(const fasttext::sent_t& sent){

  std::cout<<" words; "<<std::endl;
  for (auto& w : sent.words)
    std::cout<<"wnum "<<w.str<<" pos "<<(int)w.pos_tag
             <<" link "<<w.parent_offs<<" rel "<<(int)w.synt_rel<<std::endl;
  std::cout<<"phrases; "<<std::endl;
  for (auto& w : sent.phrases){
    std::cout<<"wnum "<<w.str<<" pos "<<(int)w.pos_tag
             <<" link "<<w.parent_offs<<" rel "<<(int)w.synt_rel
             <<" SZ "<<(int)w.sz<<std::endl;
    if (w.sz){
      std::cout<<"components: ";
      for (int i = 0; i < w.sz; i++)
        std::cout<<" "<<w.components[i];
      std::cout<<std::endl;
    }

  }

  std::cout<<"concepts; "<<std::endl;
  for (auto& s : sent.concepts)
    std::cout<<" "<<s;
  std::cout<<std::endl;

}

int main(int argc, char* argv[]){
  std::ifstream ifs(argv[1]);
  if (!ifs.is_open()){
    std::cerr<<"failed to open "<<std::endl;
    return 1;
  }

  std::string json;
  std::getline(ifs, json);
  fasttext::line_t line;
  try{
    parse_from_json(json, line);
  }catch(const std::exception& e){
    std::cerr<<"Error while parsing "<<e.what()<<std::endl;
  }

  std::cout<<"parsed stuff; "<<std::endl;
  std::cout<<"target ; "<<std::endl;
  print_sent(line.target);
  std::cout<<"other langs"<<std::endl;

  for (const auto& s : line.other_langs)
    print_sent(s);

}
