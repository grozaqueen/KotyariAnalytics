from etl.common.db import get_conn

MV = "public.mv_cluster_recency"
IDX = "mv_cluster_recency_pk"

def main():
    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {MV};")
                conn.commit()
                print("✔ Refreshed CONCURRENTLY.")
                return
            except Exception as e:
                print("[WARN] CONCURRENTLY failed:", e)

        conn.rollback()

        with conn.cursor() as cur:
            print("→ Doing initial REFRESH ...")
            cur.execute(f"REFRESH MATERIALIZED VIEW {MV};")
            conn.commit()

        with conn.cursor() as cur:
            print("→ Ensuring unique index ...")
            cur.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {IDX}
                ON {MV}(section, cluster_id);
            """)
            conn.commit()

        print("✔ Initial REFRESH done, unique index present. Next runs will use CONCURRENTLY.")

if __name__ == "__main__":
    main()
