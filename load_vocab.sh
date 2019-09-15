#/bin/bash
HOST=
PORT=
USERNAME=
PASSWORD=
INPUT_FILE_PATH=./wikipedia_vocab.csv

/opt/mlcp/bin/mlcp.sh import \
-host ${HOST} -port ${PORT} \
-username ${USERNAME} -password ${PASSWORD} \
-mode local \
-input_file_path ${INPUT_FILE_PATH} \
-input_file_type delimited_text \
-document_type json \
-output_uri_suffix ".json" \
-output_uri_prefix "/vocab/" \
-data_type "index,number,frequency,number"
