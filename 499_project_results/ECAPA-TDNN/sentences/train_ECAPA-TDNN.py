import sys
import torchaudio
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.logger import get_logger
import random
import csv
import os
import torch
import torchaudio
from tqdm.contrib import tzip
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.logger import get_logger
import re
from functools import partial
from speechbrain.dataio.dataio import read_audio
from enum import Enum
logger = get_logger(__name__)
classification = [
    "Parkinsons",
    "NonParkinsons",
]
class FileHeader(Enum):
    B = "B"
    D = "D"
    FB = "FB"
    V = "V"
logger = get_logger(__name__)
def prepare_common_status(data_folder, save_folder, skip_prep=False, file_headers:list=False):
    """
    Prepares the csv files for the CommonLanguage dataset for LID.
    Download: https://www.dropbox.com/s/qqpmqay3q9xb1vf/common_voice_kpd.tar.gz?dl=0

    Arguments
    ---------
    data_folder : str
        Path to the folder where the CommonLanguage dataset for LID is stored.
        This path should include the multi: /datasets/CommonLanguage
    save_folder : str
        The directory where to store the csv files.
    skip_prep: bool
        If True, skip data preparation.

    Returns
    -------
    None

    Example
    -------
    >>> from recipes.CommonLanguage.common_language_prepare import prepare_common_status
    >>> data_folder = '/datasets/CommonLanguage'
    >>> save_folder = 'exp/CommonLanguage_exp'
    >>> prepare_common_status(\
            data_folder,\
            save_folder,\
            skip_prep=False\
        )
    """

    if skip_prep:
        return

    # Setting the save folder
    os.makedirs(save_folder, exist_ok=True)

    # Setting output files
    save_csv_train = os.path.join(save_folder, "train.csv")
    save_csv_dev = os.path.join(save_folder, "dev.csv")
    save_csv_test = os.path.join(save_folder, "test.csv")
    """
    # If csv already exists, we skip the data preparation
    if skip(save_csv_train, save_csv_dev, save_csv_test):
        csv_exists = " already exists, skipping data preparation!"
        msg = save_csv_train + csv_exists
        logger.info(msg)
        msg = save_csv_dev + csv_exists
        logger.info(msg)
        msg = save_csv_test + csv_exists
        logger.info(msg)

        return
    """
    # Additional checks to make sure the data folder contains Common Language
    #check_common_language_folder(data_folder)

    # Audio files extensions
    extension = [".wav"]

    # Create the signal list of train, dev, and test sets.
    data_split = create_sets(data_folder, extension)
    random.shuffle(data_split["train"])
    random.shuffle(data_split["test"])
    random.shuffle(data_split["dev"])
    # Creating csv files for training, dev and test data
    create_csv(wav_list=data_split["train"], csv_file=save_csv_train)
    create_csv(wav_list=data_split["dev"], csv_file=save_csv_dev)
    create_csv(wav_list=data_split["test"], csv_file=save_csv_test)

def create_sets(data_folder, extension, file_starters:list=None):
    """
    Creates lists for train, dev and test sets with data from the data_folder

    Arguments
    ---------
    data_folder : str
        Path of the CommonLanguage dataset.
    extension: list of file extensions
        List of strings with file extensions that correspond to the audio files
        in the CommonLanguage dataset

    Returns
    -------
    dictionary containing train, dev, and test splits.
    """

    # Datasets initialization
    datasets = {"15 Young Healthy Control", "22 Elderly Healthy Control", "28 People with Parkinson's disease"}
    data_split = {dataset: [] for dataset in datasets}

    # Get the list of languages from the dataset folder
    languages = [
    "Parkinsons",
    "NonParkinsons"]

    msg = f"{len(languages)} languages detected!"
    logger.info(msg)

    # Fill the train, dev and test datasets with audio filenames
    for dataset in datasets:
            curr_folder = os.path.join(data_folder, dataset)
            wav_list = get_all_files(curr_folder, match_and=extension)
            wav_list = list(filter(partial(checkFileLocation, fileHeaders=file_starters), wav_list))
            data_split[dataset].extend(wav_list)
    msg = "Data successfully split!"
    logger.info(msg)
    machine_learing_datasets = {"train", "dev", "test"}
    machine_learing_data_split = {machine_learing_dataset: [] for machine_learing_dataset in machine_learing_datasets}
    for dataset in datasets:
          d1, d2, d3 = split_array(data_split[dataset])
          machine_learing_data_split["train"].extend(d1)
          machine_learing_data_split["dev"].extend(d2)
          machine_learing_data_split["test"].extend(d3)
    return machine_learing_data_split

def checkHeader(fileName, fileHeaders):
    for header in fileHeaders:
        if fileName.startswith(header.value):
            return True
    return False

def checkFileLocation(fileAddress, fileHeaders):
    fileName = fileAddress.split("/")[-1]
    if fileHeaders is None:
        return True
    for header in fileHeaders:
        if fileName.startswith(header.value):
            return True
    return False

def split_array(arr):
    # Shuffle the array to randomize the elements
    random.shuffle(arr)

    # Compute split sizes
    n = len(arr)
    size1 = int(n * 0.7)  # 70% of the elements
    size2 = (n - size1) // 2  # 15% of the elements
    size3 = n - size1 - size2  # Remaining 15%

    # Split the shuffled array
    part1 = arr[:size1]
    part2 = arr[size1:size1 + size2]
    part3 = arr[size1 + size2:]

    return part1, part2, part3

def create_csv(wav_list, csv_file):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    csv_file : str
        The path of the output json file
    """

    # Adding some Prints
    msg = f"Creating csv lists in {csv_file} ..."
    logger.info(msg)

    csv_lines = []

    # Start processing lines
    total_duration = 0.0

    # Starting index
    idx = 0

    for wav_file in tzip(wav_list):
        wav_file = wav_file[0]

        path_parts = wav_file.split(os.path.sep)
        file_name, wav_format = os.path.splitext(path_parts[-1])

        # Peeking at the signal (to retrieve duration in seconds)
        if os.path.isfile(wav_file):
            info = torchaudio.info(wav_file)
        else:
            msg = "\tError loading: %s" % (str(len(file_name)))
            logger.info(msg)
            continue

        audio_duration = info.num_frames / info.sample_rate
        total_duration += audio_duration
        status = None
        # Actual name of the language
        print(f"path_parts:{path_parts}")
        if  path_parts[-4] == "28 People with Parkinson's disease":
            status = "Parkinsons"
        else:
            status = "NonParkinsons"


        # Create a row with whole utterances
        csv_line = [
            idx,  # ID
            wav_file,  # File name
            wav_format,  # File format
            str(info.num_frames / info.sample_rate),  # Duration (sec)
            status,  # Language
        ]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

        # Increment index
        idx += 1

    # CSV column titles
    csv_header = ["ID", "wav", "wav_format", "duration", "status"]

    # Add titles to the list at index 0
    csv_lines.insert(0, csv_header)

    # Writing the csv lines
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = f"{csv_file} successfully created!"
    logger.info(msg)
    msg = f"Number of samples: {len(wav_list)}."
    logger.info(msg)
    msg = f"Total duration: {round(total_duration / 3600, 2)} hours."
    logger.info(msg)

"""Recipe for training a LID system with CommonStatus.

To run this recipe, do the following:
> python train.py hparams/train_ecapa_tdnn.yaml

Author
------
 * Mirco Ravanelli 2021
 * Pavlo Ruban 2021
"""

logger = get_logger(__name__)


# Brain class for Status ID training
class SID(sb.Brain):
    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.

        Returns
        -------
        feats : torch.Tensor
            Computed features.
        lens : torch.Tensor
            The length of the corresponding features.
        """
        wavs, lens = wavs

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, lens = self.hparams.wav_augment(wavs, lens)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm_input(feats, lens)

        return feats, lens

    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : torch.Tensor
            torch.Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings and output
        feats, lens = self.prepare_features(batch.sig, stage)
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, inputs, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        inputs : tensors
            The output tensors from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        predictions, lens = inputs

        targets = batch.status_encoded.data

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "wav_augment"):
                targets = self.hparams.wav_augment.replicate_labels(targets)
                if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                    self.hparams.lr_annealing.on_batch_end(self.optimizer)

        loss = self.hparams.compute_cost(predictions, targets)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, targets, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_common_status` to have been called before this,
    so that the `train.csv`, `dev.csv`,  and `test.csv` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "dev" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'lang01': 0, 'lang02': 1, ..)
    status_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig, _ = torchaudio.load(wav)
        sig = sig.transpose(0, 1).squeeze(1)
        if sig.shape[0] < 16000:  # 1 second at 16kHz
            sig = torch.nn.functional.pad(sig, (0, 16000 - sig.shape[0]))
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("status")
    @sb.utils.data_pipeline.provides("status", "status_encoded")
    def label_pipeline(status):
        yield status
        status_encoded = status_encoder.encode_label_torch(status)
        yield status_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["train", "dev", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"{dataset}_csv"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "status_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    status_encoder_file = os.path.join(
        hparams["save_folder"], "status_encoder.txt"
    )
    status_encoder.load_or_create(
        path=status_encoder_file,
        from_didatasets=[datasets["train"]],
        output_key="status",
    )

    return datasets, status_encoder


# Recipe begins!
if __name__ == "__main__":
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_common_status,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "skip_prep": hparams["skip_prep"],
            "file_headers":[FileHeader.FB, FileHeader.B],
        },
    )
    # Data preparation for augmentation
    sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])
    sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

    # Create dataset objects "train", "dev", and "test" and status_encoder
    datasets, status_encoder = dataio_prep(hparams)

    # Fetch and load pretrained modules
    #hparams["pretrainer"].collect_files()
    #hparams["pretrainer"].load_collected()

    # Initialize the Brain object to prepare for mask training.
    sid_brain = SID(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    sid_brain.fit(
        epoch_counter=sid_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["dev"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = sid_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
