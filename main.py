import cv2
import mediapipe as mp

# Inicializa o MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Inicialize a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2) as hands:
    message_displayed = False
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Converte o formato da imagem para RGB
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        binary_number = 0
        
        if results.multi_hand_landmarks:
            # Loop através de todas as mãos detectadas
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenha os landmarks da mão
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                # Obtém a posição x do pulso (landmark[0]) para distinguir entre a mão esquerda e direita
                hand_x = hand_landmarks.landmark[0].x
                if hand_x < 0.5:
                    # Mão à esquerda da tela
                    left_big_finger = hand_landmarks.landmark[4]  # Landmarks do polegar
                    left_index_finger = hand_landmarks.landmark[8]  # Landmarks do dedo indicador
                    left_middle_finger = hand_landmarks.landmark[12]  # Landmarks do dedo médio
                    left_ring_finger = hand_landmarks.landmark[16]  # Landmarks do dedo anelar
                    left_little_finger = hand_landmarks.landmark[20]  # Landmarks do dedo mínimo
                    left_index_finger_y = left_index_finger.y
                    left_middle_finger_y = left_middle_finger.y
                    left_ring_finger_y = left_ring_finger.y
                    left_little_finger_y = left_little_finger.y
                    # Calcula o número binário com base na posição dos dedos da mão esquerda
                    if left_index_finger_y < hand_landmarks.landmark[7].y: binary_number += 16
                    if left_middle_finger_y < hand_landmarks.landmark[11].y: binary_number += 32
                    if left_ring_finger_y < hand_landmarks.landmark[15].y: binary_number += 64
                    if left_little_finger_y < hand_landmarks.landmark[19].y: binary_number += 128
                else:
                    # Mão à direita da tela
                    right_big_finger = hand_landmarks.landmark[4]  # Landmarks do polegar
                    right_index_finger = hand_landmarks.landmark[8]  # Landmarks do dedo indicador
                    right_middle_finger = hand_landmarks.landmark[12]  # Landmarks do dedo médio
                    right_ring_finger = hand_landmarks.landmark[16]  # Landmarks do dedo anelar
                    right_little_finger = hand_landmarks.landmark[20]  # Landmarks do dedo mínimo
                    right_index_finger_y = right_index_finger.y
                    right_middle_finger_y = right_middle_finger.y
                    right_ring_finger_y = right_ring_finger.y
                    right_little_finger_y = right_little_finger.y
                    # Calcula o número binário com base na posição dos dedos da mão direita
                    if right_index_finger_y < hand_landmarks.landmark[7].y: binary_number += 8
                    if right_middle_finger_y < hand_landmarks.landmark[11].y: binary_number += 4
                    if right_ring_finger_y < hand_landmarks.landmark[15].y: binary_number += 2
                    if right_little_finger_y < hand_landmarks.landmark[19].y: binary_number += 1

        # Converte o número binário em uma string
        message = str(binary_number)
        message_displayed = True
        
        # Exibe a mensagem na tela
        if message_displayed:
            text_position = (50, 50)
            cv2.putText(image, message, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Exibe a imagem com os resultados
        cv2.imshow('MediaPipe Hands', image)
        
        # Verifica se o usuário pressionou a tecla Esc (código 27) para sair do loop
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Libera a captura de vídeo
cap.release()
cv2.destroyAllWindows()