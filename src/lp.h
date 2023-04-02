#ifndef FASTTEXT_LP_H
#define FASTTEXT_LP_H

#include <cstdint>

namespace fasttext{
//#SYNTAX
//# UD version 2
enum class  SyntRel : unsigned {
  ROOT       = 0,
  NSUBJ      = 1,
  OBJ        = 2,
  OBL        = 3,
  ADVMOD     = 4,
  AMOD       = 5,
  NMOD       = 6,
  CASE       = 7,
  ACL        = 8,
  CC         = 9,
  APPOS      = 10,
  COMPOUND   = 11,
  CONJ       = 12,
  DEP        = 13,
  MARK       = 14,
  NUMMOD     = 15,
  AUX        = 16,
  FLAT       = 17,
  CCOMP      = 18,
  CLF        = 19,
  COP        = 20,
  CSUBJ      = 21,
  ADVCL      = 22,
  DET        = 23,
  DISCOURSE  = 24,
  DISLOCATED = 25,
  EXPL       = 26,
  FIXED      = 27,
  GOESWITH   = 28,
  IOBJ       = 29,
  LIST       = 30,
  ORPHAN     = 31,
  PARATAXIS  = 32,
  PUNCT      = 33,
  REPARANDUM = 34,
  VOCATIVE   = 35,
  XCOMP      = 36
};

}



#endif // FASTTEXT_LP_H
