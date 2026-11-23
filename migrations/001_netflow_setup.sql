-- Earlier installers removed the active predicate and created enum columns
-- incompatible with the application's text parameters and run statuses.
ALTER TABLE model_versions DROP CONSTRAINT IF EXISTS model_versions_lane_id_is_active_key;
DROP INDEX IF EXISTS uq_one_active_model_per_lane;

DO $migration$
DECLARE
    col record;
BEGIN
    FOR col IN
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND data_type = 'USER-DEFINED'
          AND (table_name, column_name) IN (
              ('lanes', 'lane_id'), ('taxonomy_map', 'lane_id'),
              ('ingestion_jobs', 'lane_id'), ('model_versions', 'lane_id'),
              ('ingestion_jobs', 'status'), ('runs', 'status'), ('incidents', 'severity')
          )
    LOOP
        EXECUTE format('ALTER TABLE public.%I ALTER COLUMN %I DROP DEFAULT',
                       col.table_name, col.column_name);
        EXECUTE format('ALTER TABLE public.%I ALTER COLUMN %I TYPE text USING %I::text',
                       col.table_name, col.column_name, col.column_name);
        IF col.column_name = 'status' THEN
            EXECUTE format('ALTER TABLE public.%I ALTER COLUMN status SET DEFAULT ''queued''',
                           col.table_name);
        END IF;
    END LOOP;
END
$migration$;
CREATE UNIQUE INDEX uq_one_active_model_per_lane ON model_versions(lane_id) WHERE is_active;
