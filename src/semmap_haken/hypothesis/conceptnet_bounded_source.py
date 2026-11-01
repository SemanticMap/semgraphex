"""Prepare a bounded, reproducibly sampled ConceptNet 5.7 source.

Download of the official full 5-column assertions file is notebook-owned.
This module reservoir-samples *all eligible English relation assertions*
from a streamed gzip, not the first N rows. It preserves original TSV records.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import random

DEFAULT_RELATIONS=("RelatedTo","IsA","PartOf","UsedFor","HasProperty","HasA")


def reservoir_source(source: str | Path, destination: str | Path, *,
                     maximum: int=600_000, seed: int=42,
                     relations: tuple[str,...]=DEFAULT_RELATIONS):
    source,destination=Path(source),Path(destination)
    if not source.is_file(): raise FileNotFoundError(source)
    if destination.exists(): raise FileExistsError(destination)
    if maximum < 1 or not relations: raise ValueError("invalid reservoir settings")
    rng=random.Random(seed)
    reservoir=[]
    accepted=0
    tallies=Counter()
    opener=gzip.open if source.suffix==".gz" else open
    with opener(source,"rt",encoding="utf-8",newline="") as stream:
        for line in stream:
            parts=line.rstrip("\n").split("\t",4)
            if len(parts)!=5: continue
            if not (parts[2].startswith("/c/en/") and
                    parts[3].startswith("/c/en/")): continue
            name=parts[1].rsplit("/",1)[-1]
            if name not in relations: continue
            accepted+=1
            tallies[name]+=1
            if len(reservoir)<maximum:
                reservoir.append(line)
            else:
                j=rng.randrange(accepted)
                if j<maximum: reservoir[j]=line
    if not accepted or len(reservoir)<maximum//2:
        raise ValueError(f"too few eligible English assertions ({accepted})")
    destination.parent.mkdir(parents=True,exist_ok=True)
    pending=destination.with_name(destination.name+".partial")
    try:
        with pending.open("w",encoding="utf-8") as f:
            f.writelines(reservoir)
        os.replace(pending,destination)
    finally:
        pending.unlink(missing_ok=True)
    digest=hashlib.sha256()
    with destination.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): digest.update(chunk)
    summary={
        "policy":"uniform reservoir of eligible assertions from full source stream",
        "source":str(source),"seed":seed,"maximum":maximum,
        "eligible":accepted,"selected":len(reservoir),
        "eligible_by_relation":dict(tallies),"selected_sha256":digest.hexdigest(),
        "warning":"Assertion-level uniform subsampling, followed by degree-seeded "
                  "relation-stratified BFS; not uniform vertex sampling."
    }
    destination.with_name(destination.name+".sample.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return summary


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--maximum",type=int,default=600000)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--relations",default=",".join(DEFAULT_RELATIONS))
    x=ap.parse_args(argv)
    report=reservoir_source(x.input,x.output,maximum=x.maximum,seed=x.seed,
           relations=tuple(y.strip() for y in x.relations.split(",") if y.strip()))
    print(json.dumps(report,ensure_ascii=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
