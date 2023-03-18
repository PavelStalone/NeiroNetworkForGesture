from tflite_support import metadata as _metadata

displayer = _metadata.MetadataDisplayer.with_model_file("meta.tflite")
export_json_file = "extracted_metadata.json"
json_file = displayer.get_metadata_json()

# Optional: write out the metadata as a json file
with open(export_json_file, "w") as f:
  f.write(json_file)