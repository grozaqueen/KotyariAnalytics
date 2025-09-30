import os

import hdbscan
import psycopg2
import numpy as np
import umap
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, Normalizer, normalize
from alg.embedings_with_proxy import get_embeddings
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

load_dotenv()

def cluster_messages():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode="disable",
        connect_timeout=5,
    )

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, text, section
                FROM public.topics
            """)
            rows = cur.fetchall()

            data = []
            for row in rows:
                id_, text, section = row
                data.append({
                    'id': id_,
                    'text': text,
                    'section': section,
                    'original_text': text
                })

            texts = [item['text'] for item in data]
            embeddings = get_embeddings(texts)

            for i, item in enumerate(data):
                item['embedding'] = embeddings[i]
                print(embeddings[i], len(embeddings[i]))

            clusters = perform_clustering(data)

            visualize_clusters(data, clusters)

            save_clusters_to_db(conn, data, clusters)

            return data, clusters

    except Exception as e:
        print(f"Ошибка: {e}")
        conn.rollback()
        return [], []
    finally:
        conn.close()



def perform_clustering(data, method='umap_hdbscan', auto_eps=True, k_neighbors=5):
    embeddings = np.array([item['embedding'] for item in data])

    embeddings_normalized = normalize(embeddings, norm="l2")

    if method == 'dbscan':
        if auto_eps:
            nn = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine")
            nn.fit(embeddings_normalized)
            distances, _ = nn.kneighbors(embeddings_normalized)
            k_distances = np.sort(distances[:, k_neighbors - 1])
            plt.plot(k_distances); plt.ylabel(f"{k_neighbors}-NN distance")
            plt.xlabel("Points sorted by distance"); plt.title("Используй излом графика как eps")
            plt.show()
            eps_value = float(np.median(k_distances))
        else:
            eps_value = 0.3

        clustering = DBSCAN(eps=eps_value, min_samples=10, metric='cosine')
        labels = clustering.fit_predict(embeddings_normalized)

    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
        labels = kmeans.fit_predict(embeddings_normalized)

    elif method == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=8,
            min_samples=2,
            metric='cosine',
            cluster_selection_method='leaf',
            cluster_selection_epsilon=0.05,
            prediction_data=True
        )
        labels = clusterer.fit_predict(embeddings_normalized)

    elif method == 'umap_hdbscan':
        reducer_15 = umap.UMAP(
            n_neighbors=10,
            min_dist=0.0,
            n_components=15,
            metric='cosine',
            random_state=42
        )
        emb_umap = reducer_15.fit_transform(embeddings_normalized)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=6,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='leaf',
            cluster_selection_epsilon=0.03,
            prediction_data=True
        )
        labels = clusterer.fit_predict(emb_umap)

        reducer_2 = umap.UMAP(
            n_neighbors=10, min_dist=0.0, n_components=2,
            metric='cosine', random_state=42
        )
        emb_2d = reducer_2.fit_transform(embeddings_normalized)
        for i, item in enumerate(data):
            item['xy'] = emb_2d[i]

    else:
        raise ValueError("Метод должен быть 'dbscan', 'kmeans', 'hdbscan' или 'umap_hdbscan'")

    return labels


def find_optimal_eps(embeddings, k=2):
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    distances = np.sort(distances[:, k - 1], axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title(f'K-Distance Graph (k={k})')
    plt.ylabel(f'Distance to {k}-th nearest neighbor')
    plt.xlabel('Points sorted by distance')
    plt.grid(True)

    knee = np.diff(distances, n=2)
    optimal_eps = distances[np.argmax(knee) + 1]

    plt.axhline(y=optimal_eps, color='r', linestyle='--',
                label=f'Suggested eps: {optimal_eps:.3f}')
    plt.legend()
    plt.show()

    return optimal_eps


def improved_clustering(embeddings, auto_tune=True, min_samples=2):

    normalizer = Normalizer()
    embeddings_normalized = normalizer.fit_transform(embeddings)

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_normalized)

    if embeddings.shape[1] > 100:
        pca = PCA(n_components=0.95)
        embeddings_processed = pca.fit_transform(embeddings_scaled)
        print(f"Уменьшено размерность с {embeddings.shape[1]} до {pca.n_components_}")
    else:
        embeddings_processed = embeddings_scaled

    if auto_tune:
        eps = find_optimal_eps(embeddings_processed, k=min_samples)
        min_samples = max(2, int(0.01 * len(embeddings)))
    else:
        eps = 0.3
        min_samples = 2

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1
    )

    labels = clustering.fit_predict(embeddings_processed)

    return labels, clustering, eps, min_samples


def advanced_clustering(embeddings, method='auto'):

    processed_embeddings = Normalizer().fit_transform(embeddings)

    if method == 'auto':
        results = {}

        for metric in ['cosine', 'euclidean']:
            for eps in [0.3, 0.4, 0.5]:
                clustering = DBSCAN(eps=eps, min_samples=2, metric=metric)
                labels = clustering.fit_predict(processed_embeddings)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                results[(metric, eps)] = {
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'labels': labels
                }

        best_result = min(results.items(),
                          key=lambda x: (x[1]['n_noise'], -x[1]['n_clusters']))

        return best_result[1]['labels'], best_result[0]

    else:
        clustering = DBSCAN(eps=0.4, min_samples=3, metric='cosine')
        return clustering.fit_predict(processed_embeddings), ('cosine', 0.4)


def evaluate_clustering(embeddings, labels):

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=labels, cmap='tab20', alpha=0.7, s=30)
    plt.colorbar(scatter)
    plt.title('Результаты кластеризации')

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)

    print(f"Найдено кластеров: {n_clusters}")
    print(f"Точек шума: {n_noise} ({n_noise / len(labels) * 100:.1f}%)")
    print(f"Размеры кластеров:")

    for label in sorted(unique_labels):
        if label != -1:
            size = list(labels).count(label)
            print(f"  Кластер {label}: {size} точек")


def optimize_clustering_pipeline(data):
    texts = [item['text'] for item in data]
    embeddings = get_embeddings(texts)

    optimal_eps = 0.6
    min_samples = 15

    labels, clustering, eps_used, min_samples_used = improved_clustering(
        embeddings, auto_tune=True
    )

    evaluate_clustering(embeddings, labels)

    n_noise = list(labels).count(-1)
    if n_noise > len(labels) * 0.5:
        labels = DBSCAN(eps=eps_used * 0.8, min_samples=min_samples_used,
                        metric='cosine').fit_predict(embeddings)

    return labels


def calculate_cosine_similarity(emb1, emb2):
    return cosine_similarity([emb1], [emb2])[0][0]


def calculate_cosine_distance(emb1, emb2):
    return 1 - calculate_cosine_similarity(emb1, emb2)


def visualize_clusters(data, clusters):
    try:
        xy = [item.get('xy') for item in data]
        if all(v is not None for v in xy):
            embeddings_2d = np.vstack(xy)
        else:
            embeddings = np.array([item['embedding'] for item in data])
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                              c=clusters, cmap='viridis', alpha=0.7, s=30)
        plt.colorbar(scatter)
        plt.title('Кластеризация текстов')
        plt.xlabel('Dim 1'); plt.ylabel('Dim 2')

        for i, (x, y) in enumerate(embeddings_2d):
            if i % 10 == 0:
                plt.annotate(str(clusters[i]), (x, y), fontsize=8)

        plt.savefig('clusters_visualization.png')
        plt.show()

    except Exception as e:
        print(f"Ошибка при визуализации: {e}")


def save_clusters_to_db(conn, data, clusters):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DROP TABLE IF EXISTS public.message_clusters;
                CREATE TABLE IF NOT EXISTS public.message_clusters (
                    topic_id INTEGER PRIMARY KEY,
                    cluster_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    section TEXT NOT NULL
                )
            """)

            for i, item in enumerate(data):
                cur.execute("""
                    INSERT INTO message_clusters (topic_id, cluster_id, text, section)
                    VALUES (%s, %s, %s, %s)
                """, (item['id'], int(clusters[i]), item['text'], item['section']))

            conn.commit()
            print(f"Сохранено {len(data)} записей в таблицу message_clusters")

    except Exception as e:
        print(f"Ошибка при сохранении в базу: {e}")
        conn.rollback()
        raise


def find_similar_messages(target_text, data, top_n=5):
    target_embedding_all = [(item['text'], item['embedding']) for item in data]

    target_embedding = None

    for a, b in target_embedding_all:
        if a == target_text:
            target_embedding = b

    print("GOAL", target_embedding, len(target_embedding), get_embeddings([target_text]))

    similarities = []
    for item in data:
        similarity = calculate_cosine_similarity(target_embedding, item['embedding'])
        similarities.append((item['id'], item['text'], similarity))

    similarities.sort(key=lambda x: x[2], reverse=True)

    print(f"\nСамые похожие сообщения на: '{target_text}'\n")
    for i, (id_, text, sim) in enumerate(similarities[:top_n]):
        print(f"{i + 1}. ID: {id_}, Схожесть: {sim:.3f}")
        print(f"   Текст: {text[:100]}...")
        print()


if __name__ == "__main__":
    data, clusters = cluster_messages()

    cluster_counts = defaultdict(int)
    for cluster_id in clusters:
        cluster_counts[cluster_id] += 1

    print("Результаты кластеризации:")
    for cluster_id, count in cluster_counts.items():
        print(f"Кластер {cluster_id}: {count} сообщений")
