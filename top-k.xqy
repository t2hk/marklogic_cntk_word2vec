xquery version "1.0-ml";

(::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  指定されたJSON配列のトップKを求める。 
  $targets : 対象のデータをまとめたJSON配列
  $top-k : 上位何位までを求めるか
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::)
declare function local:top-k($targets as json:array, $top-k as xs:integer){
  (: top-kを求めるモデルを構築する。 :)
  let $input-size := json:array-size($targets)
  let $input-variable := cntk:input-variable(cntk:shape(($input-size)), "float")
  let $top-k := cntk:top-k($input-variable, $top-k)
  let $output-variable := cntk:function-outputs($top-k)
  
  (: 引数のJSON配列から、モデルの入力値を組み立てる。 :)
  let $input-value := cntk:batch(cntk:shape(($input-size)), $targets, cntk:gpu(0), "float")
  let $input-value-pair := json:to-array(($input-variable, $input-value))

  (: モデルを実行する。 :)
  let $result := cntk:evaluate($top-k, $input-value-pair, $output-variable , cntk:gpu(0))
  let $result-array := cntk:value-to-array($output-variable, $result)

  (: 結果は第一要素が実際の値、第2要素がそれらの配列インデックスである。 :)
  return ($result-array[1], $result-array[2])
};

(: 数値10～19のトップ5を算出する例である。
   結果は以下のように、実際の値とその配列インデックスである。
   [[19, 18, 17, 16, 15]] : 実際の値
   [[9, 8, 7, 6, 5]] : 上記値の配列インデックス
:)
let $result := local:top-k(json:to-array((10 to 19)), 5)
return $result