import tensorflow as tf
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import os
import time
from tensorflow import keras
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_path = os.path.join('App', 'models', 'model.h5')
labelencoder_path = os.path.join('App', 'models', 'labelencoder.npy')

model = tf.keras.models.load_model(model_path)
labelencoder = LabelEncoder()
labelencoder.classes_ = np.load(labelencoder_path, allow_pickle=True)

insect_info = {
    "Bee": {
        "image": "images/bee.jpg",
        "description": "Bees are flying insects closely related to wasps and ants.\n Role: Crucial pollinators for many plants and crops.\n Honey Production: Honeybees produce honey from nectar.\n Social Structure: Many bees live in colonies with a queen, workers, and drones..\n Communication: Honeybees use the waggle dance to share information.\n"
    },
    "Beetle": {
        "image": "images/beetle.jpg",
        "description": "Beetles are insects with hard exoskeletons and forewings called elytra, known for their immense diversity and roles as decomposers, pollinators, and predators.\n Role: Key decomposers and pollinators, essential for ecosystems.\n Diet: Includes plants, other insects, and decomposing matter.\n Habitat: Found almost everywhere.\n Diversity: Includes well-known types like ladybugs, fireflies, and dung beetles.\n"
    },
    "Bess Beetle": {
        "image": "images/bess_beetle.jpg",
        "description": "Bess beetles are a type of beetle known for their hard bodies and social behavior.\n Role: Important decomposers, breaking down wood and recycling nutrients in forest ecosystems.\n Traits: Hard exoskeleton and strong mandibles for chewing wood.\n Habitat: Found in decaying wood, particularly in forests.\n Diet: Feeds on decaying wood and fungi within it.\n"
    },
    "Bumblebee": {
        "image": "images/bumblebee.jpg",
        "description": "Bumblebees are robust flying insects closely related to honeybees.\n Role:Bumblebees play a crucial role in ecosystems and agriculture by pollinating plants and contributing to biodiversity, making their conservation essential for sustainable food production and natural habitats.\n Characteristics: Bumblebees are large, fuzzy insects with round bodies covered in soft hair.\n Behavior: Bumblebees are social insects that live in colonies with a queen, workers, and drones.\n Habitat: They are found primarily in temperate climates, but some species can also be found in higher altitudes or in tropical regions.\n"
    },
    "Butterfly": {
        "image": "images/butterfly.jpg",
        "description": "Butterflies are colorful flying insects known for their delicate wings and important role in pollination.\n Role: Important pollinators contributing to ecosystem health and biodiversity.\n Species: Belong to the order Lepidoptera, with numerous species worldwide.\n Habitat: Found in diverse habitats, including gardens, meadows, and forests.\n Traits: Delicate wings covered in scales, colorful patterns for camouflage and signaling.\n Diet: Adults feed on flower nectar; caterpillars feed on host plants.\n"
    },
    "Click Beetle": {
        "image": "images/click_beetle.jpg",
        "description": "Click beetles, scientifically known as Elateridae, are a family of beetles characterized by their ability to flip themselves into the air with an audible clicking sound when placed on their backs.\n Role: Click beetles are important decomposers in ecosystems, helping to break down organic matter and recycle nutrients.\n Traits: They are characterized by a unique mechanism that allows them to click and flip into the air when placed on their backs, aiding in escaping predators.\n Habitat: Click beetles are found in diverse habitats including forests, fields, and gardens.\n Diet: As larvae, they feed on decaying plant material and roots. Adults primarily feed on nectar and pollen.\n "
    },
    "Cockroach": {
        "image": "images/cockroach.jpg",
        "description": "Cockroaches are known for their resilience and adaptability, thriving in various environments worldwide.\n Role: Cockroaches play a role in ecosystems as decomposers, breaking down organic matter. However, they are often considered pests due to their presence in human habitats, where they can spread diseases and trigger.\n Habitat: Found in various environments, particularly warm and humid areas such as kitchens, bathrooms, and basements.\n Diet: Omnivorous scavengers that feed on organic matter, including food scraps, decaying material, and even glue and paper.\n Traits: They are characterized by their flattened bodies, long antennae, and a pair of wings that may be either fully developed or reduced.\n"
    },
    "Cricket": {
        "image": "images/cricket.jpg",
        "description": "Crickets are insects belonging to the family Gryllidae, known for their distinctive chirping sounds, which males produce by rubbing their wings together to attract females. \n Role: Crickets are important for nutrient cycling, soil health, and serve as prey for many predators.\n Habitat: Found in various habitats including fields, forests, gardens, and sometimes inside buildings.\n Traits: Crickets have cylindrical bodies, long antennae.\n Diet: Omnivorous, feeding on plants, fungi, and small insects.\n"
    },
    "Dor Beetle": {
        "image": "images/dor_beetle.jpg",
        "description": "Dor beetles dig tunnels beneath dung piles to lay their eggs, which provides a food source for their larvae.\n Role: Important decomposers, aiding in nutrient recycling and soil aeration through their burrowing activities.\n Traits: They are stout, dark-colored beetles known for their burrowing behavior.\n Habitat: Found in forests, grasslands, and agricultural areas.\n Species: Dor beetles belong to the family Geotrupidae, with several species.\n"
    },
    "Dragonfly": {
        "image": "images/dragonfly.jpg",
        "description": "Dragonflies are a common subject in art and symbolism, often representing change and transformation.\n Role: Important predators of insect populations, helping control pests and contributing to the health of aquatic ecosystems.\n Diet: Carnivorous, feeding on other insects, including mosquitoes and flies.\n Habitat: Found near freshwater habitats like ponds, lakes, and rivers.\n Traits: Dragonflies have elongated bodies, large multifaceted eyes, and two pairs of strong, transparent wings.\n "
    },
    "Dung Beetle": {
        "image": "images/dung_beetle.jpg",
        "description": "Dung beetles are divided into three main groups based on their behavior: rollers, tunnelers, and dwellers.\n Rollers: These beetles shape dung into balls and roll them to a suitable location for burial. This behavior is often associated with mating, as males offer the dung balls to females as a form of courtship.\n Tunnelers: These beetles dig tunnels beneath dung piles and bury pieces of dung within these tunnels to provide food and a nesting site for their larvae.\n Dwellers: These beetles live directly in dung piles and lay their eggs there, where the larvae develop.\n Role: Important decomposers, aiding in nutrient recycling, soil aeration, and reducing parasitic loads in environments.\n"
    },
    "Grasshopper": {
        "image": "images/grasshopper.jpg",
        "description": "Grasshoppers can have significant ecological impacts. In large numbers, they can become pests, causing substantial damage to crops and vegetation.\n Role: Important in food chains as prey for many animals, and can influence plant community dynamics through their feeding.\n Traits: They have long hind legs adapted for jumping, and short antennae.\n Habitat: Found in a variety of habitats including grasslands, forests, and agricultural fields.\n Species: Grasshoppers belong to the suborder Caelifera, with over 11,000 species worldwide.\n"
    },
    "House Fly": {
        "image": "images/house_fly.jpg",
        "description": "House flies can carry and transmit various pathogens, including bacteria, viruses, and parasites, due to their feeding and breeding habits.\n Role: Known as pests due to their potential to spread diseases, but also play a role in decomposition and nutrient recycling.\n Traits: Small, with a gray thorax, four dark longitudinal stripes on the back, and slightly hairy bodies.\n Habitat: Found worldwide, especially in areas with human activity such as homes, farms, and garbage dumps.\n Diet: Omnivorous, feeding on decaying organic matter, food waste, and animal feces.\n"
    },
    "Katydid": {
        "image": "images/katydid.jpg",
        "description": "Katydid males produce characteristic mating calls by rubbing their wings together, a behavior known as stridulation. These sounds are often described as katy-did or katy-didn't, which is how they got their common name.\n Role: Important in ecosystems as both prey and consumers of plant material, contributing to the balance of plant and insect populations.\n Traits: Known for their leaf-like appearance, long antennae.\n Diet: Mostly herbivorous, feeding on leaves, flowers, and stems; some species are omnivorous.\n Habitat: Found in various habitats including forests, grasslands, and gardens.\n"
    },
    "Locust": {
        "image": "images/locust.jpg",
        "description": "Locusts are a type of short-horned grasshopper and distinguished by their ability to undergo dramatic behavioral and physiological changes in response to environmental conditions, particularly crowding.\n Role: Can have a significant impact on agriculture due to their swarming behavior, causing extensive crop damage and affecting food security.\n Traits: Normally solitary, but can form large, destructive swarms under certain conditions; have powerful hind legs for jumping and flying.\n Habitat: Found in various environments, including grasslands, deserts, and agricultural areas.\n Diet: Herbivorous, feeding on a wide range of plants and crops.\n"
    },
    "Long Horn Beetle": {
        "image": "images/long_horn_beetle.jpg",
        "description": "Longhorn beetles are of interest to scientists and entomologists due to their diverse behaviors, life cycles, and interactions with their environments.\n Role: Important decomposers, aiding in the breakdown of dead wood and recycling nutrients in forest ecosystems.\n Traits: Characterized by their long antennae, often longer than their bodies, and elongated bodies.\n Diet: Larvae feed on wood, while adults feed on leaves, nectar, and bark.\n Habitat: Found in forests, woodlands, and areas with abundant dead or decaying wood.\n "
    },
    "Mole Cricket": {
        "image": "images/mole_cricket.jpg",
        "description": "Mole crickets are insects belonging to the family Gryllotalpidae within the order Orthoptera.\n Role: Known for aerating soil through their burrowing activities, but can also be pests due to their feeding on plant roots and damaging crops.\n Traits: Known for their burrowing behavior, they have strong forelimbs adapted for digging and are typically brown or tan.\n Habitat: Found in moist soil environments, including lawns, gardens, and agricultural fields.\n Diet: Omnivorous, feeding on roots, tubers, and other plant material, as well as small insects and larvae.\n"
    },
    "Mosquito": {
        "image": "images/mosquito.jpg",
        "description": "Mosquitoes are small, flying insects known for their ability to feed on the blood of animals, including humans, using specialized mouthparts called proboscis.\n Role: While some species serve as pollinators, females are known for their role as vectors of diseases such as malaria, dengue fever, Zika virus, and West Nile virus, posing significant health risks to humans and animals alike.\n Traits: Small flying insects with slender bodies, long legs, and a characteristic proboscis for feeding.\n Habitat: Found in various habitats worldwide, especially where standing water is present, such as marshes, ponds, and urban areas.\n Diet: Females feed on blood to obtain proteins necessary for egg development; males feed on nectar and other plant fluids.\n"
    },
    "Moth": {
        "image": "images/moth.jpg",
        "description": "Moths belong to the order Lepidoptera, which also includes butterflies. They are diverse insects known for their typically dull-colored wings, although some species are brightly colored or patterned.\n Role: Moths play important roles in ecosystems as pollinators for many plants, contributors to food webs as prey for birds and other animals, and some species are economically important as pests of crops and stored products.\n Traits: Typically have thick bodies, feathery or filamentous antennae (compared to the clubbed antennae of butterflies), and usually rest with their wings spread out.\n Habitat: Found in diverse habitats worldwide, including forests, fields, and urban areas.\n Diet: As larvae (caterpillars), moths feed on a variety of plant materials; as adults, they primarily feed on nectar.\n"
    },
    "Wasp": {
        "image": "images/wasp.jpg",
        "description": "Wasps are flying insects belonging to the order Hymenoptera, which also includes bees and ants.\n Role: Wasps play important roles in ecosystems as predators, helping to control insect populations. Some species are also pollinators, although to a lesser extent than bees. However, some wasps can be pests, particularly those that build nests near human habitation and can sting in defense.\n Traits: Typically have slender bodies with a narrow waist, often brightly colored or marked with distinct patterns.\n Habitat: Found in various habitats worldwide, including gardens, forests, and urban areas.\n Diet: Most adult wasps are predators, feeding on other insects and spiders; some also feed on nectar and sweet substances.\n"
    },
        
}

def augment_audio(audio, sample_rate):
    pitch_shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=np.random.uniform(-2, 2))
    time_stretched = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
    noise = np.random.randn(len(audio))
    noise_added = audio + 0.005 * noise
    return [pitch_shifted, time_stretched, noise_added]


def feature_extractor(audio, sample_rate):
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

def record_audio(duration=6, fs=22050):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    audio = audio.flatten()
    return audio, fs

def classify_sound(audio, sample_rate):
    mfcc_features = feature_extractor(audio, sample_rate)
    mfccs_scaled_features = mfcc_features.reshape(1, -1)
    result_array = model.predict(mfccs_scaled_features)
    result_index = np.argmax(result_array)
    confidence = result_array[0][result_index]
    if confidence < 0.5:
        return "Non-bug Sound", confidence
    else:
        return labelencoder.inverse_transform([result_index])[0], confidence
 

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/classify', methods=['POST'])
def classify():
    recorded_audio, recorded_sample_rate = record_audio(duration=6)
    augmented_audios = augment_audio(recorded_audio, recorded_sample_rate)

    # Classify each augmented audio
    results = []
    confidences = []

    for augmented_audio in augmented_audios:
        result, confidence = classify_sound(augmented_audio, recorded_sample_rate)
        results.append(result)
        confidences.append(confidence)

    # Take the result with the highest confidence
    max_confidence_index = np.argmax(confidences)
    result = results[max_confidence_index]
    confidence = confidences[max_confidence_index]

    insect_data = insect_info.get(result, {"image": "", "description": "Unknown insect"})

    return render_template('result.html', results=[(result, confidence)], insect_data=[insect_data])


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(temp_file)
        
        try:
            sound_signal, sample_rate = librosa.load(temp_file, res_type="kaiser_fast")
            mfcc_features = feature_extractor(sound_signal, sample_rate)
            mfccs_scaled_features = mfcc_features.reshape(1, -1)
            
            result_array = model.predict(mfccs_scaled_features)
            result_index = np.argmax(result_array)
            confidence = result_array[0][result_index]
            
            if confidence < 0.5:
                result = "Non-bug Sound"
            else:
                result = labelencoder.inverse_transform([result_index])[0]

            insect_data = insect_info.get(result, {"image": "", "description": "Unknown insect"})

            return render_template('result.html', results=[(result, confidence)], insect_data=[insect_data])

        except Exception as e:
            print(f"Error processing file: {e}")
            return 'Error processing file'

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)



if __name__ == '__main__':
    app.run(debug=True)


