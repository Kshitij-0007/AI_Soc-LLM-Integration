UPDATE model_versions
SET model_path = 'models/netflow_model/' || substr(replace(model_path, chr(92), '/'), 16)
WHERE lane_id = 'netflow'
  AND starts_with(replace(model_path, chr(92), '/'), 'models/netflow/');
