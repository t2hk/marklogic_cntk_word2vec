xquery version "1.0-ml";

(::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  単語間のCosine距離を取得する。
  バッチ処理に対応しており、各引数の配列のインデックスが同じ単語同士の距離をまとめて算出する。
  $source-words : 比較元の単語インデックスのJSON配列
  $target-words : 比較先の単語インデックスのJSON配列  
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::)
declare function local:cosine-distance($source-words as json:array, $target-words as json:array){
  (: 当該モデルで学習した単語数 :)
  let $VOCAB_SIZE := 345861
  
  (:学習データの準備:)
  let $model := cntk:function(fn:doc("/model/wikipedia_w2v_model.onnx")/binary(), cntk:gpu(0), "onnx")
  
  (: モデルの入力パラメータ :)
  let $source-input-variable := cntk:input-variable(cntk:shape((1)), "float")
  let $target-input-variable := cntk:input-variable(cntk:shape((1)), "float")

  (: 単語インデックスをOne-Hot表現に変換するレイヤー:)
  let $source-onehot := cntk:one-hot-op($source-input-variable, $VOCAB_SIZE, fn:false(), cntk:axis(0))
  let $target-onehot := cntk:one-hot-op($target-input-variable, $VOCAB_SIZE, fn:false(), cntk:axis(0))

  (: 埋め込みレイヤーの定義。 :)
  let $emb-input-variable := cntk:input-variable(cntk:shape(($VOCAB_SIZE)), "float")
  let $emb-map := map:map()
  let $__ := map:put($emb-map, "weight", cntk:constant-value(cntk:function-constants($model)))
  let $source-emb-layer := cntk:embedding-layer($source-onehot, $emb-map)
  let $target-emb-layer := cntk:embedding-layer($target-onehot, $emb-map)
  
  (: コサイン距離を算出するモデル :)
  let $cosine-model := cntk:cosine-distance($source-emb-layer, $target-emb-layer)

  (: モデルの出力パラメータ型 :)
  let $output-variable := cntk:function-output($cosine-model)
    
  (: 入力されたデータからバッチ処理用にまとめたデータを組み立てる。:)
  let $source-values := for $i in (1 to json:array-size($source-words)) return json:to-array(($source-words[$i]))
  let $target-values := for $i in (1 to json:array-size($target-words)) return json:to-array(($target-words[$i]))
  let $continues := for $i in (1 to json:array-size($source-words)) return fn:true()
  
  let $source-bos := cntk:batch-of-sequences(cntk:shape((1)), json:to-array(($source-values)), $continues, cntk:gpu(0))
  let $target-bos := cntk:batch-of-sequences(cntk:shape((1)), json:to-array(($target-values)), $continues, cntk:gpu(0))
  let $source-input-pair := json:to-array(($source-input-variable, $source-bos))  
  let $target-input-pair := json:to-array(($target-input-variable, $target-bos))  
  
  let $input-pair := json:to-array(($source-input-pair, $target-input-pair))

  (: 単語のベクトル表現を取得する:)
  let $result := cntk:evaluate($cosine-model, $input-pair, $output-variable, cntk:gpu(0))  
  
  return cntk:value-to-array($output-variable, $result) 
};

(: 単語インデックス10～19の10単語と、100～109の10単語、それぞれのCosine距離を求める例 
   単語10と単語100、単語11と単語101・・・の距離である。:)
let $result = local:cosine-distance(json:to-array((10 to 19))), json:to-array((100 to 109)))
return $result
