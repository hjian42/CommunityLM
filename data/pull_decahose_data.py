import datetime
import boto3
import pyathena

conn = pyathena.connect(aws_access_key_id="XXX",
                        aws_secret_access_key="XXX",
                        # schema_name='lexical_framing',
                        s3_staging_dir='s3://lsm-data/ccc-hang/athena_tables/',
                        region_name='us-east-1')

date_starts = [
    '2019-01-01','2019-02-01','2019-03-01','2019-04-01','2019-05-01','2019-06-01',
    '2019-07-01','2019-08-01','2019-09-01','2019-10-01','2019-11-01','2019-12-01',
    '2020-01-01','2020-02-01','2020-03-01','2020-04-01','2020-05-01','2020-06-01', '2020-07-01'
]

cursor = conn.cursor()

for start_day_str,end_day_str in zip(date_starts[:-1],date_starts[1:]):
    print(f"Processing date range: [{start_day_str},{end_day_str}). Running query at time: {datetime.datetime.now()}")
    
    query1 = f"""
        CREATE TABLE IF NOT EXISTS lexical_framing.us_deca_{start_day_str.replace('-','')}
        WITH (
            format='PARQUET',
            write_compression = 'SNAPPY',
            external_location='s3://lsm-data/lexical-framing/us_deca/us_deca_{start_day_str.replace('-','')}/',
            partitioned_by = ARRAY['day'])
        AS
        SELECT
          "split_part"("json_extract_scalar"(line, '$.id'), ':', 3) id
        , "json_extract_scalar"(line, '$.postedTime') postedTime
        , "json_extract_scalar"(line, '$.verb') verb

        , "split_part"("json_extract_scalar"(line, '$.actor.id'), ':', 3) actor_id
        , "json_extract_scalar"(line, '$.actor.preferredUsername') actor_preferredUsername
        , "json_extract_scalar"(line, '$.actor.displayName') actor_displayName
        , "json_extract_scalar"(line, '$.actor.summary') actor_summary
        , "json_extract_scalar"(line, '$.actor.image') actor_image
        , CAST("json_extract_scalar"(line, '$.actor.friendsCount') AS INTEGER) actor_friendsCount
        , CAST("json_extract_scalar"(line, '$.actor.followersCount') AS INTEGER) actor_followersCount
        , CAST("json_extract_scalar"(line, '$.actor.statusesCount') AS INTEGER) actor_statusesCount
        , CAST("json_extract_scalar"(line, '$.actor.favoritesCount') AS INTEGER) actor_favoritesCount
        , "json_extract_scalar"(line, '$.actor.location.displayName') actor_location_displayName

        , "json_extract_scalar"(line, '$.twitter_lang') twitter_lang
        , "json_extract_scalar"(line, '$.body') body
        , "json_extract_scalar"(line, '$.long_object.body') longobj_body
        , "json_extract_scalar"(line, '$.object.long_object.body') object_longobj_body
        , day
        FROM
          decahose.decahose_line
        WHERE (
            (line <> '') AND 
            ("json_extract_scalar"(line, '$.objectType') = 'activity') AND
            ("json_extract_scalar"(line, '$.verb') = 'post') AND
            ("json_extract_scalar"(line, '$.twitter_lang') = 'en') AND
            (("json_extract_scalar"(line, '$.actor.location.displayName') <> '') OR 
                ("json_extract_scalar"(line, '$.actor.location.displayName') IS NOT NULL)) AND
            (day >= DATE '{start_day_str}' AND day < DATE '{end_day_str}')
            )
    """
    cursor.execute(query1)
    print(start_day_str, end_day_str, "Successful!")
