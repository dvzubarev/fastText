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
  ADVCL      = 9,
  APPOS      = 10,
  AUX        = 11,
  CC         = 12,
  CCOMP      = 13,
  CLF        = 14,
  COMPOUND   = 15,
  CONJ       = 16,
  COP        = 17,
  CSUBJ      = 18,
  DEP        = 19,
  DET        = 20,
  DISCOURSE  = 21,
  DISLOCATED = 22,
  EXPL       = 23,
  FIXED      = 24,
  FLAT       = 25,
  GOESWITH   = 26,
  IOBJ       = 27,
  LIST       = 28,
  MARK       = 29,
  NUMMOD     = 30,
  ORPHAN     = 31,
  PARATAXIS  = 32,
  PUNCT      = 33,
  REPARANDUM = 34,
  VOCATIVE   = 35,
  XCOMP      = 36
};

}



#endif // FASTTEXT_LP_H
