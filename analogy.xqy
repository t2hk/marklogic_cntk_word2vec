xquery version "1.0-ml";

(:単語のインデックス値を取得する。:)
declare function local:word-to-index($word as xs:string){
  let $uris :=
  cts:uris((), (), 
    cts:and-query((
      cts:directory-query("/vocab/", "infinity"),
      cts:json-property-value-query("word", $word))
    ))
  return xs:float(fn:doc($uris)/index/number())[1]
};

(:インデックスに対応する単語を取得する。:)
declare function local:index-to-word($index as xs:integer){
  let $uris :=
  cts:uris((), (), 
    cts:and-query((
      cts:directory-query("/vocab/", "infinity"),
      cts:json-property-value-query("index", $index))
    ))
  return fn:doc($uris)/word[1]
};

(: 類推 :)
declare function local:analogy($a-word as xs:string, $b-word as xs:string, $c-word as xs:string, $top-k as xs:integer){
  (: 当該モデルで学習した単語数 :)
  let $VOCAB_SIZE := 345861
  
  (:学習データの準備:)
  let $model := cntk:function(fn:doc("/model/wikipedia_w2v_model.onnx")/binary(), cntk:gpu(0), "onnx")
  
  (: モデルの入力パラメータ型 :)
  let $a-input-variable := cntk:input-variable(cntk:shape((1)), "float")
  let $b-input-variable := cntk:input-variable(cntk:shape((1)), "float")
  let $c-input-variable := cntk:input-variable(cntk:shape((1)), "float")

  (:単語をOne-Hot表現に変換するレイヤー:)
  let $a-onehot := cntk:one-hot-op($a-input-variable, $VOCAB_SIZE, fn:false(), cntk:axis(0))
  let $b-onehot := cntk:one-hot-op($b-input-variable, $VOCAB_SIZE, fn:false(), cntk:axis(0))
  let $c-onehot := cntk:one-hot-op($c-input-variable, $VOCAB_SIZE, fn:false(), cntk:axis(0))

  (: 埋め込みレイヤーの定義 :)
  let $emb-input-variable := cntk:input-variable(cntk:shape(($VOCAB_SIZE)), "float")
  let $emb-map := map:map()
  let $__ := map:put($emb-map, "weight", cntk:constant-value(cntk:function-constants($model)))
  let $a-emb-layer := cntk:embedding-layer($a-onehot, $emb-map)
  let $b-emb-layer := cntk:embedding-layer($b-onehot, $emb-map)
  let $c-emb-layer := cntk:embedding-layer($c-onehot, $emb-map)
  
  (:単語ベクトルの加減算のレイヤー定義:)
  let $b-min-a := cntk:minus($b-emb-layer, $a-emb-layer)
  let $b-min-a-plus-c := cntk:plus($b-min-a, $c-emb-layer )
  let $normalize := cntk:element-divide($b-min-a-plus-c, cntk:sqrt(cntk:reduce-sum-on-axes(cntk:element-times($b-min-a-plus-c, $b-min-a-plus-c), cntk:axis(0))))
  
  (: モデルの出力パラメータ型 :)
  let $output-variable := cntk:function-outputs($normalize)
  
  (: 入力データとその定義 :)
  let $a-val := json:to-array((local:word-to-index($a-word)))
  let $b-val := json:to-array((local:word-to-index($b-word)))
  let $c-val := json:to-array((local:word-to-index($c-word)))

  let $a-input-value := cntk:batch(cntk:shape((1)), $a-val, cntk:gpu(0), "float")
  let $b-input-value := cntk:batch(cntk:shape((1)), $b-val, cntk:gpu(0), "float")
  let $c-input-value := cntk:batch(cntk:shape((1)), $c-val, cntk:gpu(0), "float")  

  let $a-input-value-pair := json:to-array(($a-input-variable, $a-input-value))
  let $b-input-value-pair := json:to-array(($b-input-variable, $b-input-value))
  let $c-input-value-pair := json:to-array(($c-input-variable, $c-input-value))
  let $input-value-pair := json:to-array((  $a-input-value-pair, $b-input-value-pair, $c-input-value-pair))

  (: 単語ベクトルの加減算を実行 :)
  let $result := cntk:evaluate($normalize, $input-value-pair, $output-variable, cntk:gpu(0))
  let $rv := cntk:value-to-array($output-variable, $result)
  
  (: 最も近い単語を検索する :)
  let $analogy_word := local:most-similar($rv, $a-val, $b-val, $c-val, $top-k)
  let $results-cos-distance := $analogy_word[1]
  let $results-index := json:array-values($analogy_word[2])[1]
  let $results-words :=
    for $i in (1 to json:array-size($results-index))
      return local:index-to-word(xs:integer($results-index[$i]))
      
  return($results-cos-distance, $results-words)  
};

let $result := local:analogy("イタリア", "ローマ", "フランス", 5)
return $result
