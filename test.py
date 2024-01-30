from psycopg2 import Error
import pandas as pd
from sqlalchemy import create_engine

def insert_from_csv(path):
    try:
        flag = False
        columns = ["guid", "eye", "id_patient", "image", "feature"]
        df = pd.read_csv(path, sep="\t", names=columns)
        engine = create_engine('postgresql://xgb_pupil:La-sRzbcQ3DE@postgres81.1gb.ru:5432/xgb_pupil')
        df.to_sql('features', engine, index=False, if_exists='append')
        flag = True
    except (Exception, Error) as err:
        print("Ошибка копирования таблицы:", err)
    finally:
        return flag

print(insert_from_csv('table.csv'))