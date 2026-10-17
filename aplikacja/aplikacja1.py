import cv2
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

# --- KONFIGURACJA STAŁA ---
TARGET_WIDTH = 800

# Słownik Braille'a (Standard Polski standard 2x3)
BRAILLE_DICT = {
    (1, 0, 0, 0, 0, 0): 'a', (1, 1, 0, 0, 0, 0): 'b', (1, 0, 0, 1, 0, 0): 'c',
    (1, 0, 0, 1, 1, 0): 'd', (1, 0, 0, 0, 1, 0): 'e', (1, 1, 0, 1, 0, 0): 'f',
    (1, 1, 0, 1, 1, 0): 'g', (1, 1, 0, 0, 1, 0): 'h', (0, 1, 0, 1, 0, 0): 'i',
    (0, 1, 0, 1, 1, 0): 'j', (1, 0, 1, 0, 0, 0): 'k', (1, 1, 1, 0, 0, 0): 'l',
    (1, 0, 1, 1, 0, 0): 'm', (1, 0, 1, 1, 1, 0): 'n', (1, 0, 1, 0, 1, 0): 'o',
    (1, 1, 1, 1, 0, 0): 'p', (1, 1, 1, 1, 1, 0): 'q', (1, 1, 1, 0, 1, 0): 'r',
    (0, 1, 1, 1, 0, 0): 's', (0, 1, 1, 1, 1, 0): 't', (1, 0, 1, 0, 0, 1): 'u',
    (1, 1, 1, 0, 0, 1): 'v', (0, 1, 0, 1, 1, 1): 'w', (1, 0, 1, 1, 0, 1): 'x',
    (1, 0, 1, 1, 1, 1): 'y', (1, 0, 1, 0, 1, 1): 'z',
    # Polskie znaki diakrytyczne
    (1, 0, 0, 0, 0, 1): 'ą', (1, 1, 0, 1, 0, 1): 'ć', (1, 0, 0, 1, 0, 1): 'ę',
    (1, 1, 1, 0, 1, 1): 'ł', (1, 1, 0, 1, 1, 1): 'ń', (0, 1, 0, 1, 0, 1): 'ó',
    (1, 1, 1, 1, 0, 1): 'ś', (1, 1, 0, 0, 1, 1): 'ź', (0, 1, 1, 1, 0, 1): 'ż',
    # Znaki specjalne i interpunkcja
    (0, 0, 0, 0, 0, 0): ' ', (0, 0, 1, 1, 1, 1): '#', (0, 0, 0, 0, 0, 1): '^',
    (0, 1, 0, 0, 0, 0): ',', (0, 1, 1, 0, 0, 0): ';', (0, 1, 0, 0, 1, 0): ':',
    (0, 1, 0, 0, 1, 1): '.', (0, 1, 0, 0, 0, 1): '?', (0, 1, 1, 0, 0, 1): '!',
    (0, 0, 1, 0, 1, 1): '-',
} 
BRAILLE_DIGITS = {
    'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5',
    'f': '6', 'g': '7', 'h': '8', 'i': '9', 'j': '0'
}

def load_and_resize_image(image_path, target_width):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Nie można wczytać obrazu: {image_path}")
    scale = target_width / img.shape[1]
    target_height = int(img.shape[0] * scale)
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

def align_image_by_dots(image_roi, dots_with_area):
    """
    Poprawiona funkcja wyrównywania: Rygorystyczne klastrowanie poziome
    zapobiega fałszywemu dopasowywaniu linii na ukos.
    """
    if len(dots_with_area) < 6:  # Potrzebujemy kilku kropek, by wyznaczyć stabilną linię
        return image_roi, dots_with_area

    dots_np = np.array([[d[0], d[1]] for d in dots_with_area])
    avg_dist = calculate_avg_dot_distance(dots_with_area)
    dots_scaled = dots_np.copy()
    dots_scaled[:, 1] = dots_scaled[:, 1] * 2.5  # Sztucznie rozciągamy pion
    # Epsilon ustawiony bardziej restrykcyjnie, aby łączyć tylko bliskie sąsiedztwo w poziomie
    # Zamiast 2.5, używamy około 1.6-1.8 odległości bazowej
    db = DBSCAN(eps=avg_dist * 1.8, min_samples=3).fit(dots_scaled)
    labels = db.labels_
    
    angles = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:
            continue
        
        # Do cv2.fitLine przekazujemy ORYGINALNE (nieskalowane) punkty z danego klastra
        cluster_points = dots_np[labels == label]
        # Interesują nas tylko długie, wyraźne linie (min. 5 kropek)
        if len(cluster_points) < 5:
            continue
            
        [vx, vy, x, y] = cv2.fitLine(cluster_points, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = np.degrees(np.arctan2(vy, vx))[0]
        
        # Normalizacja do zakresu (-45, 45)
        if angle > 45: angle -= 90
        elif angle < -45: angle += 90
        # Interesują nas małe korekty (Braille rzadko bywa obrócony o więcej niż 25 stopni)
        if -25 < angle < 25:
            angles.append(angle)
        
    if not angles:
        return image_roi, dots_with_area
        print("[WYRÓWNANIE] Brak wiarygodnych linii do wyznaczenia obrotu. Pomijam.")
    # Odrzucamy skrajne wartości (outliery) używając mediany
    median_angle = np.median(angles)
    
    if abs(median_angle) < 0.3:
        print(f"[WYRÓWNANIE] Kąt minimalny ({median_angle:.2f}°). Obraz jest prosty.")
        return image_roi, dots_with_area
    print(f"[WYRÓWNANIE] Wykryty stabilny kąt linii: {median_angle:.2f}°")
    
    # Obrót ROI
    h, w = image_roi.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated_roi = cv2.warpAffine(image_roi, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Transformacja punktów
    # ones = np.ones(shape=(len(dots_np), 1))
    # points_ones = np.hstack((dots_np, ones))
    # transformed_points = M.dot(points_ones.T).T
    dots_reshaped = dots_np.reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv2.transform(dots_reshaped, M).squeeze()
    new_dots = []
    for i, pt in enumerate(transformed_points):
        new_dots.append([int(pt[0]), int(pt[1]), dots_with_area[i][2]])
    
    return rotated_roi, new_dots

def get_adaptive_params(image_roi):
    h, w = image_roi.shape[:2]
    base_dim = np.sqrt(h * w)  
    
    blur_sz = int(base_dim * 0.015)
    if blur_sz % 2 == 0: blur_sz += 1
    blur_sz = max(3, blur_sz)
    
    block_sz = int(base_dim * 0.025)
    if block_sz % 2 == 0: block_sz += 1
    block_sz = max(5, block_sz)
    
    c_val = 5 
    morph_sz = int(base_dim * 0.005)
    morph_sz = max(2, morph_sz)
    return (blur_sz, blur_sz), block_sz, c_val, (morph_sz, morph_sz)
    
def preprocess_image_adaptive(image, blur_kernel, block_size, c_val, morph_kernel):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, block_size, c_val
    )
    kernel = np.ones(morph_kernel, np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed
    
def detect_dots(contours):
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4: 
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if 0.3 < circularity < 1.6:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                candidates.append([cX, cY, area])
                
    if not candidates:
        return []

    areas = [c[2] for c in candidates]
    median_area = np.median(areas)

    auto_min_area = median_area * 0.35
    auto_max_area = median_area * 2.8
    
    dots = [c for c in candidates if auto_min_area < c[2] < auto_max_area]
    print(f"Auto-progowanie pola kropki: Min={auto_min_area:.1f}, Max={auto_max_area:.1f} (Mediana: {median_area:.1f})")
    return dots
    
def calculate_avg_dot_distance(dots):
    if len(dots) < 2:
        return 20
    dots_coords = np.array([[dot[0], dot[1]] for dot in dots])
    tree = KDTree(dots_coords)
    distances, _ = tree.query(dots_coords, k=2)
    distances = distances[:, 1]
    
    filtered_distances = distances[distances < np.percentile(distances, 80)]
    if len(filtered_distances) == 0:
        return 20
    return np.median(filtered_distances)

def decode_braille_pipeline(dots_with_area):
    upper_case_next = False  
    number_mode = False
    if not dots_with_area: 
        return "", [], 20

    dots_np = np.array([[d[0], d[1]] for d in dots_with_area])
    avg_dist = calculate_avg_dot_distance(dots_with_area)
    print(f"Automatycznie wyznaczona odległość bazowa kropek (avg_dist): {avg_dist:.2f} px")
    
    # 1. SEGMENTACJA NA LINIE TEKSTU
    y_coords = dots_np[:, 1].reshape(-1, 1)
    db_lines = DBSCAN(eps=avg_dist * 1.5, min_samples=1).fit(y_coords)
    line_labels = db_lines.labels_
    
    unique_lines = sorted(list(set(line_labels)))
    line_medians = [np.median(dots_np[line_labels == l_id, 1]) for l_id in unique_lines]
    sorted_lines = [x for _, x in sorted(zip(line_medians, unique_lines))]
    
    decoded_lines = []
    visualized_characters = []
    
    for line_id in sorted_lines:
        line_dots = dots_np[line_labels == line_id]
        
        y_line_coords = line_dots[:, 1].reshape(-1, 1)
        db_rows = DBSCAN(eps=avg_dist * 0.6, min_samples=1).fit(y_line_coords)
        row_labels = db_rows.labels_
        unique_rows = sorted(list(set(row_labels)))
        row_medians = sorted([np.median(line_dots[row_labels == r_id, 1]) for r_id in unique_rows])
        # Zapewniamy mapowanie do linii bazowej
        line_min_y = row_medians[0] if len(row_medians) > 0 else np.min(line_dots[:, 1])

        # 2. SEGMENTACJA NA ZNAKI W LINII
        x_coords = line_dots[:, 0].reshape(-1, 1)
        db_chars = DBSCAN(eps=avg_dist * 1.2, min_samples=1).fit(x_coords)
        char_labels = db_chars.labels_
        
        unique_chars = sorted(list(set(char_labels)))
        char_means = [np.mean(line_dots[char_labels == c_id, 0]) for c_id in unique_chars]
        sorted_chars = [x for _, x in sorted(zip(char_means, unique_chars))]
        
        line_text = []
        prev_cell_col0 = None
        
        for char_id in sorted_chars:
            char_dots = line_dots[char_labels == char_id]
            
            char_xs = char_dots[:, 0]
            min_cx = np.min(char_xs)
            max_cx = np.max(char_xs)
            
            if max_cx - min_cx > avg_dist * 0.5:
                cell_col0 = min_cx
            else:
                cell_col0 = min_cx
                if prev_cell_col0 is not None:
                    dist = min_cx - prev_cell_col0
                    if abs(dist - 1 * avg_dist) < abs(dist - 2.5 * avg_dist) and (max_cx - min_cx <= avg_dist * 0.3):
                        cell_col0 = min_cx - avg_dist
            
            # Detekcja spacji między znakami
            if prev_cell_col0 is not None:
                char_distance = cell_col0 - prev_cell_col0
                if char_distance > avg_dist * 3.2:  
                    line_text.append(' ')
            
            # Mapowanie punktów na siatkę 2x3
            grid = [0] * 6
            for dot in char_dots:
                # Dopasowanie do najbliższego rzędu z wykrytych median
                if len(row_medians) > 1:
                    row = np.argmin([abs(dot[1] - ry) for ry in row_medians])
                else:
                    row = int(np.round((dot[1] - line_min_y) / avg_dist))
                
                col = int(np.round((dot[0] - cell_col0) / avg_dist))
                
                row = max(0, min(2, row))
                col = max(0, min(1, col))
                
                pos = row + (col * 3)
                grid[pos] = 1
               
            letter = BRAILLE_DICT.get(tuple(grid), '?')
            
            if letter == '^':
                upper_case_next = True
                prev_cell_col0 = cell_col0
                continue  # Pomijamy dodawanie samego znaku '^' do tekstu wyjściowego
            
            if letter == '#':
                number_mode = True
                prev_cell_col0 = cell_col0
                continue

            if number_mode:
                if letter in BRAILLE_DIGITS:
                    letter = BRAILLE_DIGITS[letter]
                elif letter == ' ':
                    number_mode = False
                else:
                    number_mode = False            
                
            if upper_case_next:
                letter = letter.upper()
                upper_case_next = False
               
            line_text.append(letter)

            visualized_characters.append({
                'bbox': (int(np.min(char_dots[:, 0])), int(np.min(char_dots[:, 1])),
                         int(np.max(char_dots[:, 0])), int(np.max(char_dots[:, 1]))),
                'text': letter
            })
            
            prev_cell_col0 = cell_col0
        decoded_lines.append("".join(line_text))
    return "\n".join(decoded_lines), visualized_characters, avg_dist
 
def visualize(image, dots_with_area, visualized_characters):
    output = image.copy()
    
    for dot in dots_with_area:
        cv2.circle(output, (int(dot[0]), int(dot[1])), 3, (0, 255, 0), -1)
    
    for vc in visualized_characters:
        x1, y1, x2, y2 = vc['bbox']
        p = 6  
        pt1 = (max(0, x1 - p), max(0, y1 - p))
        pt2 = (min(image.shape[1], x2 + p), min(image.shape[0], y2 + p))
        
        cv2.rectangle(output, pt1, pt2, (255, 0, 0), 2)
        cv2.putText(output, vc['text'], (pt1[0], pt1[1] - 5), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return output
# --- GŁÓWNA CZĘŚĆ SKRYPTU ---
if __name__ == "__main__":
    try:
        img_scaled = load_and_resize_image('brail5.jpg', TARGET_WIDTH)
        
        print("Zaznacz obszar tekstu Braille'a i naciśnij ENTER...")
        roi_box = cv2.selectROI("Wybierz obszar zainteresowania (ROI)", img_scaled, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Wybierz obszar zainteresowania (ROI)")
        
        x, y, w, h = [int(v) for v in roi_box]
        img_roi = img_scaled[y:y+h, x:x+w] if (w > 0 and h > 0) else img_scaled.copy()
        
        # 1. Detekcja kropek na pierwotnym, potencjalnie skrzywionym ROI
        blur_k, block_sz, c_val, morph_k = get_adaptive_params(img_roi)
        processed = preprocess_image_adaptive(img_roi, blur_k, block_sz, c_val, morph_k)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dots_initial = detect_dots(contours)
        
        print(f"Liczba wstępnie wykrytych kropek: {len(dots_initial)}")
        
        if dots_initial:
            # 2. Prostowanie ROI za pomocą nowej metody i transformacja punktów
            img_roi_aligned, dots_aligned = align_image_by_dots(img_roi, dots_initial)
            
            # 3. Dekodowanie z prostych kropek na prostym ROI
            decoded_text, visualized_characters, avg_dist = decode_braille_pipeline(dots_aligned)
            
            print(f"\nLiczba zidentyfikowanych znaków tekstowych: {len(visualized_characters)}")
            print(f"\nZdekodowany tekst:\n{decoded_text}\n")
            
            # Wizualizacja na obróconym obrazie
            output = visualize(img_roi_aligned, dots_aligned, visualized_characters)
        else:
            print("Nie wykryto żadnych kropek do zdekodowania.")
            output = img_roi
        
        cv2.imshow("Wykryte znaki Braillea (Zrównane ROI)", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    except Exception as e:
        print(f"Błąd: {e}")