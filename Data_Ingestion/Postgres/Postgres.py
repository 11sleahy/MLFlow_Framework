import pandas as pd
import numpy as np
import sqlalchemy as sql
import psycopg2


cred = {'host':'***.rds.amazonaws.com',
        'port':'5432',
        'db':'***',
        'user':'***',
        'password': '***'}

def enter_credentials(host,user,password,port,database):
    url = f'postgresql://{user}:{password}@{host}:{port}/{database}'
    output = sql.create_engine(url)
    return output

get_output = enter_credentials(cred['host'],
                               cred['user'],
                               cred['password'],
                               cred['port'],
                               cred['db'])


dt = pd.read_sql_query(
    """
    with dupe_tmp as (select 
id,admin_id,data ->> 'score' as score,data ->> 'score_v1' as score_v1,
data ->> 'score_version' as score_version, 
association_ids ->> 'original_id' as original_id,
association_ids ->> 'duplicate_id' as duplicate_id,response -> 'data' -> 'attributes' ->> 'should_merge' as should_merge,
assigned_at,completed_at,created_at,updated_at
from tasks where type = 'PlaceDuplicatesReviewTask'
order by created_at asc
limit 10000)
select dupe_tmp.*,
p1.id as id_1,p1.name as name_1,p1.created_at as created_at_1,
p1.updated_at as updated_at_1,p1.address as address_1,
p1.business_category_id as business_category_id_1,
p1.data ->> 'latitude' as latitude_1,p1.data ->> 'longitude' as longitude_1,
p1.confidence as confidence_1,
p2.id as id_2,p2.name as name_2,p2.created_at as created_at_2,
p2.updated_at as updated_at_2,p2.address as address_2,
p2.business_category_id as business_category_id_2,
p2.data ->> 'latitude' as latitude_2,p2.data ->> 'longitude' as longitude_2,
p2.confidence as confidence_2
from dupe_tmp left join places p1 on 
dupe_tmp.original_id::integer = p1.id::integer
left join places p2 on 
dupe_tmp.duplicate_id::integer = p2.id::integer
""",get_output
)