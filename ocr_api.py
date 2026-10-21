import os
import time
import requests
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

INPUT_FOLDER = "/home/radhakrishna/ws/adapt_dr/data_queries/data"

OUTPUT_FOLDER = "/home/radhakrishna/ws/adapt_dr/data_queries/output"


SARVAM_OUTPUT = os.path.join(
    OUTPUT_FOLDER,
    "sarvam_output"
)

SURYA2_OUTPUT = os.path.join(
    OUTPUT_FOLDER,
    "surya2_output"
)

# ============================================================
# OCR API
# ============================================================

OCR_URL = "http://10.4.16.82:8058/ocr/test"

LANGUAGE = "hi"
MODALITY = "printed"

# ============================================================
# MODELS
# ============================================================

MODELS = {
    "Sarvam": {
        "version": "sarvam",
        "output": SARVAM_OUTPUT,
    },
    "Surya2": {
        "version": "surya2",
        "output": SURYA2_OUTPUT,
    },
}

# ============================================================
# SETTINGS
# ============================================================

REQUEST_TIMEOUT = 300

# Number of retries when API returns an error
MAX_RETRIES = 3

# Wait between retries
RETRY_DELAY = 5

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for model in MODELS.values():
    os.makedirs(
        model["output"],
        exist_ok=True
    )

# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)

# ============================================================
# GET ALL IMAGES
# ============================================================

images = sorted(
    [
        f
        for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
)

# ============================================================
# BASIC INFO
# ============================================================

print("=" * 80)
print("OCR BATCH PROCESSING")
print("=" * 80)

print(f"Input folder : {INPUT_FOLDER}")
print(f"Output folder: {OUTPUT_FOLDER}")
print(f"OCR API      : {OCR_URL}")
print(f"Language     : {LANGUAGE}")
print(f"Modality     : {MODALITY}")
print(f"Total images : {len(images)}")

print("=" * 80)


# ============================================================
# MIME TYPE
# ============================================================

def get_mime_type(filename):

    ext = os.path.splitext(
        filename
    )[1].lower()

    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(
        ext,
        "image/jpeg"
    )


# ============================================================
# OUTPUT PATH
# ============================================================

def get_output_path(
    image_name,
    output_folder
):

    base_name = os.path.splitext(
        image_name
    )[0]

    return os.path.join(
        output_folder,
        base_name + ".txt"
    )


# ============================================================
# EXTRACT OCR TEXT
# ============================================================

def extract_text(result):

    # --------------------------------------------------------
    # Case:
    #
    # [
    #     {"text": "..."}
    # ]
    # --------------------------------------------------------

    if isinstance(result, list):

        if (
            result
            and isinstance(
                result[0],
                dict
            )
        ):

            return result[0].get(
                "text",
                ""
            )

        return ""

    # --------------------------------------------------------
    # Case:
    #
    # {
    #     "text": "..."
    # }
    # --------------------------------------------------------

    if isinstance(result, dict):

        text = result.get(
            "text",
            ""
        )

        if text:
            return text

        # ----------------------------------------------------
        # Case:
        #
        # {
        #     "results": [
        #         {"text": "..."}
        #     ]
        # }
        # ----------------------------------------------------

        results = result.get(
            "results"
        )

        if (
            isinstance(results, list)
            and results
            and isinstance(
                results[0],
                dict
            )
        ):

            return results[0].get(
                "text",
                ""
            )

    return ""


# ============================================================
# RUN ONE OCR REQUEST
# ============================================================

def run_ocr(
    image_name,
    output_folder,
    version
):

    image_path = os.path.join(
        INPUT_FOLDER,
        image_name
    )

    output_path = get_output_path(
        image_name,
        output_folder
    )

    mime_type = get_mime_type(
        image_name
    )

    # ========================================================
    # RETRIES
    # ========================================================

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            # -----------------------------------------------
            # OPEN IMAGE
            # -----------------------------------------------

            with open(
                image_path,
                "rb"
            ) as image_file:

                files = {
                    "images": (
                        image_name,
                        image_file,
                        mime_type
                    )
                }

                data = {
                    "language": LANGUAGE,
                    "modality": MODALITY,
                    "version": version,
                }

                # -------------------------------------------
                # REQUEST
                # -------------------------------------------

                response = requests.post(
                    OCR_URL,
                    data=data,
                    files=files,
                    timeout=REQUEST_TIMEOUT,
                )

            # -----------------------------------------------
            # HTTP ERROR
            # -----------------------------------------------

            if not response.ok:

                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

                if attempt < MAX_RETRIES:

                    time.sleep(
                        RETRY_DELAY
                    )

                    continue

                return False, last_error

            # -----------------------------------------------
            # JSON
            # -----------------------------------------------

            try:

                result = response.json()

            except Exception:

                last_error = (
                    "Invalid JSON response: "
                    + response.text[:500]
                )

                if attempt < MAX_RETRIES:

                    time.sleep(
                        RETRY_DELAY
                    )

                    continue

                return False, last_error

            # -----------------------------------------------
            # TEXT
            # -----------------------------------------------

            text = extract_text(
                result
            )

            # -----------------------------------------------
            # SAVE
            # -----------------------------------------------

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as output_file:

                output_file.write(
                    text or ""
                )

            return True, output_path

        except requests.exceptions.Timeout:

            last_error = (
                "Request timeout"
            )

        except requests.exceptions.ConnectionError as e:

            last_error = (
                f"Connection error: {e}"
            )

        except Exception as e:

            last_error = (
                f"{type(e).__name__}: {e}"
            )

        # ====================================================
        # RETRY
        # ====================================================

        if attempt < MAX_RETRIES:

            time.sleep(
                RETRY_DELAY
            )

    return False, last_error


# ============================================================
# PROCESS ONE MODEL
# ============================================================

def process_model(
    model_name,
    version,
    output_folder
):

    print("\n")
    print("=" * 80)
    print(f"{model_name.upper()}")
    print("=" * 80)

    # ========================================================
    # FIND ONLY MISSING FILES
    # ========================================================

    pending_images = []

    already_done = 0

    for image_name in images:

        output_path = get_output_path(
            image_name,
            output_folder
        )

        if os.path.isfile(
            output_path
        ):

            already_done += 1

        else:

            pending_images.append(
                image_name
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        f"Already completed : {already_done}"
    )

    print(
        f"Remaining         : {len(pending_images)}"
    )

    # ========================================================
    # NOTHING TO DO
    # ========================================================

    if not pending_images:

        print(
            f"✓ {model_name}: "
            "All images already processed."
        )

        return {
            "processed": [],
            "failed": [],
            "skipped": already_done,
        }

    # ========================================================
    # PROCESS ONLY PENDING IMAGES
    # ========================================================

    processed = []
    failed = []

    for image_name in tqdm(
        pending_images,
        desc=f"{model_name} OCR",
        unit="image"
    ):

        success, result = run_ocr(
            image_name,
            output_folder,
            version
        )

        if success:

            tqdm.write(
                f"✓ {image_name}"
            )

            processed.append(
                image_name
            )

        else:

            tqdm.write(
                f"✗ FAILED: "
                f"{image_name} | {result}"
            )

            failed.append(
                image_name
            )

    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    print("\n")
    print(
        f"{model_name} summary:"
    )

    print(
        f"  Already done : {already_done}"
    )

    print(
        f"  Processed    : {len(processed)}"
    )

    print(
        f"  Failed       : {len(failed)}"
    )

    return {
        "processed": processed,
        "failed": failed,
        "skipped": already_done,
    }


# ============================================================
# RUN ALL MODELS
# ============================================================

results = {}

for model_name, config in MODELS.items():

    results[model_name] = process_model(
        model_name,
        config["version"],
        config["output"]
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 80)
print("FINAL OCR REPORT")
print("=" * 80)

print(
    f"Total input images: {len(images)}"
)

for model_name, result in results.items():

    print(f"\n{model_name}")

    print(
        f"  Already done : "
        f"{result['skipped']}"
    )

    print(
        f"  Processed    : "
        f"{len(result['processed'])}"
    )

    print(
        f"  Failed       : "
        f"{len(result['failed'])}"
    )

# ============================================================
# FAILED FILES
# ============================================================

for model_name, result in results.items():

    if result["failed"]:

        print("\n")
        print(
            f"{model_name} FAILED FILES"
        )

        print("-" * 80)

        for image_name in result["failed"]:

            print(
                image_name
            )

# ============================================================
# OUTPUT LOCATIONS
# ============================================================

print("\n")
print("=" * 80)
print("OUTPUT LOCATIONS")
print("=" * 80)

print(
    f"Sarvam : {SARVAM_OUTPUT}"
)

print(
    f"Surya2 : {SURYA2_OUTPUT}"
)

print("\nDONE.")
