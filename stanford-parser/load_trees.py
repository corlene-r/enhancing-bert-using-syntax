from typing import TypeVar, Callable, Any
from nltk.tree.tree import Tree
from tqdm import tqdm

class Context:
    def __init__(self, text: str, pos: int = 0, res: Any = None):
        self.text = text
        self.pos = pos
        self.res = res

    def matches(self, s: str) -> bool:
        if (self.pos >= len(self.text)) and (self.text[self.pos:self.pos + len(s)] == s):
            self.res = s
            self.pos += len(s)
            return True
        return False

    def fail(self, reset: int):
        self.pos = reset
        self.res = None

def seq(*callables):
    def seq(cxt: "Context"):
        #trace print(f"seq {cxt.pos}")
        save = cxt.pos
        res = []
        for callable in callables:
            callable(cxt)
            if cxt.res is None:
                cxt.pos = save
                cxt.res = None
                return
            res.append(cxt.res)
        cxt.res = res
    return seq

def one_of(*callables):
    def one_of(cxt: "Context"):
        #trace print(f"one_of {cxt.pos}")
        for callable in callables:
            callable(cxt)
            if cxt.res is not None:
                return
        cxt.res = None
    return one_of

def zero_or_more(callable):
    def zero_or_more(cxt: "Context"):
        #trace print(f"zero_or_more {cxt.pos}")
        res = []
        prev = cxt.pos
        while True:
            callable(cxt)
            #trace print(f"In zero_or_more {cxt.pos} {cxt.res}")
            if cxt.res is None:
                cxt.pos = prev
                cxt.res = res
                return
            prev = cxt.pos
    return zero_or_more

def call_matches(callable: Callable[[str], bool]):
    def call_matches(cxt: "Context"):
        #trace print(f"call_matches {cxt.pos}")
        if (cxt.pos >= len(cxt.text)) or (callable(cxt.text[cxt.pos]) == False):
            cxt.res = None
            return

        #trace print(f"Here! {cxt.text[cxt.pos]}")
        cxt.res = cxt.text[cxt.pos]
        cxt.pos += 1
    return call_matches

def matches(s: str):
    def matches(cxt: "Context"):
        #trace print(f"matches {cxt.pos} {s} == {cxt.text[cxt.pos:cxt.pos + len(s)]}")
        save = cxt.pos
        for c in s:
            if (cxt.pos >= len(cxt.text)) or (cxt.text[cxt.pos] != c):
                cxt.res = None
                cxt.pos = save
                return

            cxt.pos += 1

        cxt.res = s
    return matches

def is_not(callable):
    def is_not(cxt: "Context"):
        #trace print(f"is_not {cxt.pos}")
        save = cxt.pos
        callable(cxt)
        cxt.pos = save
        if cxt.res is None:
            cxt.res = True # dummy boolean value to succeed when child failed
        else:
            cxt.res = None # failed to parse because child parsed successfully
    return is_not

def join(callable1, callable2):
    def join(cxt: "Context"):
        #trace print(f"join {cxt.pos}")
        save = cxt.pos
        res = []
        callable1(cxt)
        if cxt.res is None:
            cxt.res = res
            cxt.pos = save
            return

        res.append(cxt.res)
        save = cxt.pos

        while True:
            callable2(cxt)
            if cxt.res is None:
                cxt.res = res
                cxt.pos = save
                return

            callable1(cxt)
            if cxt.res is None:
                cxt.res = res
                cxt.pos = save
                return

            res.append(cxt.res)
            save = cxt.pos

    return join

def span_of(callable):
    def span_of(cxt: "Context"):
        #trace print(f"span_of {cxt.pos}")
        start = cxt.pos
        callable(cxt)
        if cxt.res is None:
            cxt.res = None
            cxt.pos = start
            return
        cxt.res = cxt.text[start:cxt.pos]
    return span_of

def map(callable, mapper):
    def map(cxt: "Context"):
        #trace print(f"map {cxt.pos}")
        callable(cxt)
        if cxt.res is None: return
        cxt.res = mapper(cxt.res)
    return map

def rec(callable):
    def rec(cxt: "Context"):
        callable()(cxt)
    return rec

def load_trees_from(file: str) -> list[Tree]:
    with open(file, 'r', encoding='utf8') as f:
        lines = f.read().split('\n')

    def w(cxt: Context):
        #trace print(f'w {cxt.pos}')
        while True:
            if (cxt.pos >= len(cxt.text)) or (not cxt.text[cxt.pos].isspace()):
                break
            cxt.pos += 1
        cxt.res = ""

    def paren():
        return map(
            seq(
                matches('('),
                w,
                join(
                    one_of(string(), rec(paren)),
                    seq(w, matches(','), w),
                ),
                w,
                matches(')'),
            ),
            lambda x: Tree(x[2][0], x[2][1:])
        )

    def string():
        return one_of(
            map(matches('"""'), lambda _: '"'),
            map(matches('"\\"'), lambda _: '\\'),
            map(seq(matches('"'), call_matches(lambda _: True), matches('\\"')), lambda x: x[1]),
            map(
                seq(
                    matches('"'),
                    span_of(zero_or_more(
                        seq(
                            is_not(matches('"')),
                            one_of(
                                matches('\\"'),
                                call_matches(lambda _: True),
                            )
                        )
                    )),
                    matches('"'),
                ),
                lambda x: x[1]
            )
        )

    trees = []
    cxt = Context("")
    parser = seq(w, paren())
    for line in tqdm(lines, desc=f"Parsing Lines into trees for {file}"):
        if len(line) == 0:
            continue

        cxt.text = line
        cxt.pos = 0
        cxt.res = None
        parser(cxt)
        if cxt.res is None:
            print(f'\nFailed to parse line: "{line}"')
            return
        
        trees.append(cxt.res)

    #print(trees)

load_trees_from("../data/original/train_trees.tsv")