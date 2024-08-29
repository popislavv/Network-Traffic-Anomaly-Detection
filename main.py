import tensorflow as tf
import numpy as np

# Simulacija podataka o mrežnom saobraćaju
network_traffic = [
    {"source": "192.168.1.1", "destination": "192.168.1.2", "data": 'Request', 'label': 0},
    {"source": "192.168.1.2", "destination": "192.168.1.3", "data": 'Response', 'label': 0},
    # Dodajte više normalnog saobraćaja
    {"source": "192.168.1.3", "destination": "192.168.1.4", "data": 'Request', 'label': 0},
    {"source": "192.168.1.4", "destination": "192.168.1.5", "data": 'Response', 'label': 0},
    {"source": "192.168.1.7", "destination": "192.168.1.8", "data": 'Request', 'label': 0},
    {"source": "192.168.1.8", "destination": "192.168.1.9", "data": 'Response', 'label': 0},
    # Dodajte više normalnog i anomalnog saobraćaja po potrebi
    {"source": "192.168.1.5", "destination": "192.168.1.6", "data": 'Request', 'label': 1},
    {"source": "192.168.1.6", "destination": "192.168.1.7", "data": 'Response', 'label': 1},
]

# Predobrada podataka za TensorFlow model
preprocessed_data = [
    np.concatenate([np.array([int(x) for x in packet["source"].split('.')]),
                    np.array([int(x) for x in packet["destination"].split('.')]),
                    np.array([0 if packet["data"] == 'Request' else 1])])
    for packet in network_traffic
]

# Konvertovanje oznake u numerički niz
labels = np.array([packet["label"] for packet in network_traffic])

# Provera da li postoji dovoljno jedinstvenih izvora i odredišta za obuku modela
unique_sources = np.unique([packet["source"] for packet in network_traffic])
unique_destinations = np.unique([packet["destination"] for packet in network_traffic])

if len(unique_sources) < 2 or len(unique_destinations) < 2:
    print('Potrebno je najmanje dva različita izvora i odredišta za obuku modela.')
    exit(1)

# Podela podataka na trening i test skupove
split_index = int(0.8 * len(preprocessed_data))
training_data = preprocessed_data[:split_index]
testing_data = preprocessed_data[split_index:]
training_labels = labels[:split_index]
testing_labels = labels[split_index:]

# Konstruisanje složenijeg modela
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(9,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Obuka modela
model.fit(np.array(training_data), training_labels, epochs=20, verbose=0)

# Evaluacija modela na testnom skupu
evaluation = model.evaluate(np.array(testing_data), testing_labels, verbose=0)

print(f'Gubitak na testnim podacima: {evaluation[0]}')
print(f'Tačnost na testnim podacima: {evaluation[1]}')

# Detekcija anomalija na osnovu treniranog modela
predictions = model.predict(np.array(testing_data)).flatten()

# Eksperimentišite sa različitim vrednostima praga
threshold = 0.2
detected_anomalies_indices = [i for i in range(len(testing_data)) if predictions[i] < threshold]
detected_anomalies = [network_traffic[i + split_index] for i in detected_anomalies_indices]

# Prikazivanje rezultata
if detected_anomalies:
    print(f'Detektovane anomalije ({len(detected_anomalies)}):', detected_anomalies)
else:
    print('Nema detektovanih anomalija. Mrežni saobraćaj je normalan.')


