def try_model():
    model_types = ['cyto2', 'cyto3', 'nuclei', 'tissuenet_cp3']

    for model_type in model_types:
        print(f"\n🔍 Testando modello: {model_type}")
        model = models.CellposeModel(gpu=False, model_type=model_type)

        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

        for img_name in image_files:
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"⚠️ Immagine non valida, salto: {img_name}")
                continue

            masks, flows, styles = model.eval(img, channels=[0,0], diameter=None) 
            mask = masks

            # mask_out = os.path.join(output_dir, os.path.splitext(img_name)[0] + model_type + "_mask.png")
            # cv2.imwrite(mask_out, mask.astype(np.uint16))
            # === 1) maschera colorata ===
            mask_color = plt.get_cmap("nipy_spectral")(mask.astype(np.float32) / mask.max())
            mask_color = (mask_color[:, :, :3] * 255).astype(np.uint8)
            mask_color_out = os.path.join(output_dir, os.path.splitext(img_name)[0] + f"_{model_type}_maskCOLOR.png")
            cv2.imwrite(mask_color_out, cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR))

            # === 2) immagine + maschera (overlay) ===
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img_rgb = cv2.cvtColor(img_norm.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            overlay = img_rgb.copy()
            overlay_mask = (mask > 0).astype(np.uint8) * 255
            overlay[:, :, 0] = np.maximum(overlay[:, :, 0], overlay_mask)
            overlay_out = os.path.join(output_dir, os.path.splitext(img_name)[0] + f"_{model_type}_overlay.png")
            cv2.imwrite(overlay_out, overlay)


            print(f"✅ Salvata maschera per {img_name} in {mask_color_out}")

            # plt.figure(figsize=(6,6))
            # plt.imshow(img, cmap='gray')
            # plt.imshow(mask, cmap='nipy_spectral', alpha=0.5)
            # plt.title(f"{img_name} - Maschera sovrapposta")
            # plt.axis('off')
            # plt.show(block=False)
            # plt.pause(10000)
            # plt.close()

    print("🎉 Segmentazione completata!")

