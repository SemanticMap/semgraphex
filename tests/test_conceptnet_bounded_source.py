"""Full-stream ConceptNet bounded reservoir, no prefix sampling."""
import gzip
import json
from semmap_haken.hypothesis.conceptnet_bounded_source import reservoir_source
from semmap_haken.conceptnet import stream_assertions


def test_bounded_source_not_first_lines(tmp_path):
    path=tmp_path/"conceptnet.csv.gz"
    with gzip.open(path,"wt",encoding="utf-8") as f:
        for i in range(100):
            f.write(
                f"/a/{i}\t/r/{'RelatedTo' if i%2 else 'IsA'}\t"
                f"/c/en/w{i}\t/c/en/w{i+1}\t"
                +json.dumps({"weight":1.0})+"\n"
            )
        for i in range(10):
            f.write(f"/a/fr{i}\t/r/IsA\t/c/fr/chat\t/c/fr/animal\t"
                    +json.dumps({"weight":1.0})+"\n")
    output=tmp_path/"subset.tsv"
    result=reservoir_source(path,output,maximum=30,seed=42)
    rows=list(stream_assertions(output))
    assert result["eligible"]==100
    assert result["selected"]==30
    assert len(rows)==30
    assert max(int(x.assertion_uri[3:]) for x in rows)>65
    assert all(x.start_uri.startswith("/c/en/") for x in rows)
    assert len(set(x.assertion_uri for x in rows))==30
    output2=tmp_path/"subset2.tsv"
    reservoir_source(path,output2,maximum=30,seed=42)
    assert output.read_bytes()==output2.read_bytes()
