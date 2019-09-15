#/bin/bash
HOST=
PORT=
USERNAME=
PASSWORD=
INPUT_FILE_PATH=./wikipedia_w2v_model.onnx

/opt/mlcp/bin/mlcp.sh import -host ${HOST} -port ${PORT} \
-username ${USERNAME} \
-password ${PASSWORD} \
-input_file_type documents \
-document_type binary \
-input_file_path ${INPUT_FILE_PATH} \
-output_uri_replace "${INPUT_FILE_PATH}, '/model/wiki_w2v_model.onnx'"
