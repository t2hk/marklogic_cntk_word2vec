(::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  類似する単語を求める。 
  $source-word : 対象の単語インデックス
  $top-k : 類似する単語を上位何位まで求めるか
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::)
declare function local:most-similar($source-word as xs:integer, $topK as xs:integer){
  (: 当該モデルで学習した単語数:)
  let $VOCAB_SIZE := 345861
  
  (:学習モデルのロード:)
  let $model := cntk:function(fn:doc("/model/wiki_w2v_model.onnx")/binary(), cntk:gpu(0), "onnx")
  
  (:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                   Cosine距離を算出するモデルの構築
  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::)
  
  (: Cosine距離を求めるモデルの入力パラメータ定義。入力データは比較元と比較先の単語のインデックス値である。 :)
  let $source-input-variable := cntk:input-variable(cntk:shape((1)), "float")
  let $target-input-variable := cntk:input-variable(cntk:shape((1)), "float")

  (:単語のインデックス値をOne-Hot表現に変換するための定義。:)
  let $source-onehot := cntk:one-hot-op($source-input-variable, $VOCAB_SIZE, fn:false(), cntk:axis(0))
  let $target-onehot := cntk:one-hot-op($target-input-variable, $VOCAB_SIZE, fn:false(), cntk:axis(0))

  (: 埋め込みレイヤーの定義。読み込んだモデルからWeightを取得して使用する。 :)
  let $emb-input-variable := cntk:input-variable(cntk:shape(($VOCAB_SIZE)), "float")
  let $emb-map := map:map()
  let $__ := map:put($emb-map, "weight", cntk:constant-value(cntk:function-constants($model)))
  let $source-emb-layer := cntk:embedding-layer($source-onehot, $emb-map)
  let $target-emb-layer := cntk:embedding-layer($target-onehot, $emb-map)
  
  (:Cosine距離を算出するモデルと、その出力パラメータ定義を組み立てる。:)
  let $cos-distance-model := cntk:cosine-distance($source-emb-layer, $target-emb-layer)
  let $output-variable := cntk:function-output($cos-distance-model)
 
  (:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
           引数の単語からモデルの入力値を組み立て、処理を実行する。
       バッチで処理するため、バッチサイズ分の入力値をシーケンスでまとめる。
  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::)
  (: バッチサイズを定義する。 :)
  let $BATCH_SIZE := 5000
  
  (: 比較元の単語を、比較先の全単語と同数だけリスト化する。:)
  let $source-words := for $i in (0 to ($VOCAB_SIZE)) return $source-word
  
  (: 比較先の全単語を用意する。:)
  let $target-words := for $i in (0 to $VOCAB_SIZE) return $i  

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
    let $source-values := for $i in ($start to $end) return json:to-array(($source-words[$i]))
    let $target-values := for $i in ($start to $end) return json:to-array(($target-words[$i]))
    let $continues := for $i in (1 to ($end - $start + 1)) return fn:true()
      
    let $source-bos := cntk:batch-of-sequences(cntk:shape((1)), json:to-array(($source-values)), $continues, cntk:gpu(0))
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
  let $top-k := local:top-k(json:to-array(($results)), $topK)
  return ($top-k[1], $top-k[2])
};

(:単語のインデックス値を取得する。:)
declare function local:word-to-index($word as xs:string){
  let $uris :=
  cts:uris((), (), 
    cts:and-query((
      cts:directory-query("/vocab/", "infinity"),
      cts:json-property-value-query("word", $word))
    ))
  return xs:integer(fn:doc($uris)/index/number())
};

(:インデックスに対応する単語を取得する。:)
declare function local:index-to-word($index as xs:integer){
  let $uris :=
  cts:uris((), (), 
    cts:and-query((
      cts:directory-query("/vocab/", "infinity"),
      cts:json-property-value-query("index", $index))
    ))
  return fn:doc($uris)/word
};

(::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  類似する単語を求める。 
  $source-word : 対象の単語
  $top-k : 類似する単語を上位何位まで求めるか
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::)
declare function local:most-similar-word($source-word as xs:string, $topK as xs:integer){
  let $source-word-index := local:word-to-index($source-word)
  let $results := local:most-similar($source-word-index, $topK)
  let $results-cos-distance := $results[1]
  let $results-index := json:array-values($results[2])[1]
  let $results-words :=
    for $i in (1 to json:array-size($results-index))
      return local:index-to-word(xs:integer($results-index[$i]))
      
  return($results-cos-distance, $results-words)
};


(: 単語インデックス163に類似する単語の上位5件を求める例。
   結果は以下のように、類似する単語との距離と、単語インデックスである。
   自分自身との比較も含めているため、類似の第1位は自分自身であり、Cosine距離は1となる。
   [[1, 0.5196232, 0.5111292, 0.4781487, 0.4708132]]
   [[東京, 銀座, 大阪, 浅草, 東京都]]
:)
let $result := local:most_similar-word("東京", 5)
return $result
