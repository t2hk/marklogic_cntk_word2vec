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

(: 指定されたJSON配列のトップKを求める。 :)
declare function local:top-k($targets as json:array, $top-k as xs:integer){
  let $input-size := json:array-size($targets)

  let $input-variable := cntk:input-variable(cntk:shape(($input-size)), "float")
  let $top-k := cntk:top-k($input-variable, $top-k)
  let $output-variable := cntk:function-outputs($top-k)

  let $input-value := cntk:batch(cntk:shape(($input-size)), $targets, cntk:gpu(0), "float")
  let $input-value-pair := json:to-array(($input-variable, $input-value))

  let $result := cntk:evaluate($top-k, $input-value-pair, $output-variable , cntk:gpu(0))
  let $rv := cntk:value-to-array($output-variable, $result)
  return $rv
};

(:類似する単語を検索する。:)
declare function local:most-similar($normalize as json:array, $a-val, $b-val, $c-val, $top-k as xs:integer){
  (: 当該モデルで学習した単語数:)
  let $VOCAB_SIZE := 345861
  
  (:学習モデルのロード:)
  let $model := cntk:function(fn:doc("/model/wiki_w2v_model.onnx")/binary(), cntk:gpu(0), "onnx")
  
  (:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                     Cosine距離を算出するモデルの構築
  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::)
  
  (: Cosine距離を求めるモデルの入力パラメータ定義。入力データは比較元と比較先の単語のインデックス値である。
  let $source-input-variable := cntk:input-variable(cntk:value-shape($normalize), "float")
   :)
  let $source-input-variable := cntk:input-variable(cntk:shape((300)), "float")
  let $target-input-variable := cntk:input-variable(cntk:shape((1)), "float")

  (:単語のインデックス値をOne-Hot表現に変換するための定義。:)
  let $target-onehot := cntk:one-hot-op($target-input-variable, $VOCAB_SIZE, fn:false(), cntk:axis(0))

  (: 埋め込みレイヤーの定義。読み込んだモデルからWeightを取得して使用する。 :)
  let $emb-input-variable := cntk:input-variable(cntk:shape(($VOCAB_SIZE)), "float")
  let $emb-map := map:map()
  let $__ := map:put($emb-map, "weight", cntk:constant-value(cntk:function-constants($model)))
  let $target-emb-layer := cntk:embedding-layer($target-onehot, $emb-map)
  
  (:Cosine距離を算出するモデルと、その出力パラメータ定義を組み立てる。:)
  let $cos-distance-model := cntk:cosine-distance($source-input-variable, $target-emb-layer)
  let $output-variable := cntk:function-output($cos-distance-model)
 
  (:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
           引数の単語からモデルの入力値を組み立て、処理を実行する。
       バッチで処理するため、バッチサイズ分の入力値をシーケンスでまとめる。
  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::)
  (: バッチサイズを定義する。 :)
  let $BATCH_SIZE := 5000
  
  (: 比較元の単語を、比較先の全単語と同数だけリスト化する。
  let $source-words := for $i in (0 to ($VOCAB_SIZE)) return $normalize[1]
  :)
  (: 比較先の全単語を用意する。:)
  let $target-words := 
    for $i in (0 to ($VOCAB_SIZE))
      return
        if ($i = $a-val[1]) then  0
        else if ($i = $b-val[1]) then  0
        else if ($i = $c-val[1]) then  0
        else $i

  (: バッチの実行回数を計算しておく。:)
  let $BATCH_TIMES := ($VOCAB_SIZE idiv $BATCH_SIZE ) + 1  
  
  (: バッチ処理用のデータを組み立て、モデルを実行する。:)
  let $results := 
  for $i in (1 to $BATCH_TIMES)
    (: 本バッチで処理するバッチサイズ分の単語の開始と終了位置を算出する。最後はバッチサイズ以下となる。:)
    let $start := ($i - 1) * $BATCH_SIZE + 1
    let $_end := $start + $BATCH_SIZE - 1    
    let $end := 
      if ($_end > $VOCAB_SIZE) then $VOCAB_SIZE
      else $_end

    (: バッチ処理用に入力データを組み立てる。:)
    let $source-values := for $i in ($start to $end) return json:to-array(($normalize[1]))
    let $target-values := for $i in ($start to $end) return json:to-array(($target-words[$i]))
    let $continues := for $i in (1 to ($end - $start + 1)) return fn:true()
      
    let $source-bos := cntk:batch-of-sequences(cntk:shape((300)), json:to-array(($source-values)), $continues, cntk:gpu(0))
    let $target-bos := cntk:batch-of-sequences(cntk:shape((1)), json:to-array(($target-values)), $continues, cntk:gpu(0))
    let $source-input-pair := json:to-array(($source-input-variable, $source-bos))  
    let $target-input-pair := json:to-array(($target-input-variable, $target-bos)) 
    
    let $input-pair := json:to-array(($source-input-pair, $target-input-pair))
    
    (: モデルを実行する。
       指定された単語に対する全単語のCosine距離が算出される。:)
    let $result := cntk:evaluate($cos-distance-model, $input-pair, $output-variable, cntk:gpu(0)) 
    return json:array-values(cntk:value-to-array($output-variable, $result) )
  
  (: 算出したコサイン距離のうち、上位$topKを求める。
     配列の1番目はコサイン距離、2番目は単語のインデックス値である。:)
  let $top-k := local:top-k(json:to-array(($results)), $top-k)
  return ($top-k[1], $top-k[2])
};

(: 類推 :)
declare function local:analogy($a-word as xs:string, $b-word as xs:string, $c-word as xs:string, $top-k as xs:integer){
  (: 当該モデルで学習した単語数 :)
  let $VOCAB_SIZE := fn:count(cts:uris((), (), cts:directory-query("/vocab/", "infinity") )) + 3
  
  (:学習データの準備:)
  let $model := cntk:function(fn:doc("/model/wiki_w2v_model.onnx")/binary(), cntk:gpu(0), "onnx")
  
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

  let $b-min-a := cntk:minus($b-emb-layer, $a-emb-layer)
  let $b-min-a-plus-c := cntk:plus($b-min-a, $c-emb-layer )
  
  let $normalize := cntk:element-divide($b-min-a-plus-c, cntk:sqrt(cntk:reduce-sum-on-axes(cntk:element-times($b-min-a-plus-c, $b-min-a-plus-c), cntk:axis(0))))
  
  (: モデルの出力パラメータ型 :)
  let $output-variable := cntk:function-outputs($normalize)
  
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

  let $result := cntk:evaluate($normalize, $input-value-pair, $output-variable, cntk:gpu(0))

  let $rv := cntk:value-to-array($output-variable, $result)
  
  let $analogy_word := local:most-similar($rv, $a-val, $b-val, $c-val, $top-k)

  let $results-cos-distance := $analogy_word[1]
  let $results-index := json:array-values($analogy_word[2])[1]
  let $results-words :=
    for $i in (1 to json:array-size($results-index))
      return local:index-to-word(xs:integer($results-index[$i]))
      
  return($results-cos-distance, $results-words)  
};

let $result := local:analogy("日本", "東京", "フランス", 5)
return $result

