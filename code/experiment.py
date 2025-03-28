from dataclasses import dataclass, field
from run import Run
from haven import haven_utils as hu
import pickle
from nltk.tokenize import wordpunct_tokenize
from gensim.corpora.dictionary import Dictionary
import torch
from customTextDataset import *
from transformer_model import *
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
from plotting import *
from datasets import load_dataset
import os


@dataclass
class Experiment:
    n: int
    sequence_length: int
    dataset: str
    batch_size: any
    epochs: int
    runs: list[Run] = field(default_factory=list)
    vocab_size: int = 0
    training_dataset: any = None
    test_dataset: any = None

    def run_experiment(self, plot_only, path, device):
        experiment_id = hu.hash_dict({"experiment": self})
        if plot_only == False:
            print("Run Experiment")
            full_dataset = self.process_data()
            self.vocab_size, self.training_dataset, self.test_dataset = self.tokenize_data(
                full_dataset
            )
            print("Vocabulary size: " + str(self.vocab_size))
            for run_num, run in enumerate(self.runs):
                torch.manual_seed(0)  # change torch seed here
                print("-----Run " + str(run_num + 1) + "-----")
                ### Train model
                (
                    run.model_obj,
                    run.model_num_params
                ) = self.create_model(run, device)
                print("---Training---")
                run.training_loss_values, run.test_loss_values = self.train(run, device)
            with open(
                os.path.join(path, "experiments", str(experiment_id) + ".pkl"),
                "wb",
            ) as f:
                pickle.dump({"experiment": self}, f)
            f.close()
        with open(
            os.path.join(path, "experiments", str(experiment_id) + ".pkl"), "rb"
        ) as f:
            experiment = pickle.load(f)
        f.close()
        ### Plot results
        print("Plot Experiment")
        # print(experiment["experiment"])
        plot_experiment(experiment["experiment"], path)

    def process_data(self):
        np.random.seed(0)  # change numpy seed here
        if self.dataset == "tinystories":
            full_dataset = load_dataset("roneneldan/TinyStories", split="train")["text"]
            full_dataset = np.array(full_dataset)
            sample = np.random.permutation(np.size(full_dataset))[0:2*self.n]
            full_dataset = full_dataset[sample]
            full_dataset = full_dataset.tolist()
        return full_dataset

    def tokenize_data(self, full_dataset):
        ### Tokenize data
        datasetTokens = []
        j = 0
        for _, story in enumerate(full_dataset):
            tokenized_story = wordpunct_tokenize(story)
            if len(tokenized_story) >= self.sequence_length:
                tokenized_story = tokenized_story[: self.sequence_length]
                datasetTokens.append(tokenized_story)
                j = j + 1
                if j == 2*self.n:
                    break

        vocab = Dictionary(datasetTokens)
        vocab_size = len(Dictionary(datasetTokens))

        ### Convert tokens to ID's
        datasetIDs = []
        for story in datasetTokens:
            storyID = []
            for word in story:
                storyID.append(vocab.token2id[word])
            datasetIDs.append(storyID)

        datasetIDs = torch.tensor(datasetIDs)

        tokenized_dataset = CustomTextDataset(sequence=datasetIDs)
        training_dataset = tokenized_dataset[0 : self.n]
        test_dataset = tokenized_dataset[self.n : 2*self.n]

        return vocab_size, training_dataset, test_dataset

    def create_model(self, run, device):
        model = DecoderOnlyTransformer(
            omega=self.vocab_size,
            d=run.d,
            m=run.m,
            tao=self.sequence_length,
            device=device,
        ).to(device)
        summary(model)

        model_num_params = sum(p.numel() for p in model.parameters())
        return model, model_num_params

    def train(self, run, device):
        if self.batch_size == "full":
            batch_size = self.n
        else:
            batch_size = self.batch_size

        criterion = nn.CrossEntropyLoss(reduction="sum")
        step_size = 0.0001
        optimizer = optim.Adam(run.model_obj.parameters(), lr=step_size)
        ### Run Training loop
        trainloader = torch_data.DataLoader(
            self.training_dataset, batch_size=batch_size, shuffle=False
        )
        training_loss_vals = []
        test_loss_vals = []
        for epoch in range(self.epochs):
            for _, sequence_batch in enumerate(trainloader):
                sequence_batch = sequence_batch.to(device)
                optimizer.zero_grad()
                output = run.model_obj(sequence_batch[:, :-1])
                loss = criterion(
                    output.contiguous().view(-1, self.vocab_size),
                    sequence_batch[:, 1:].contiguous().view(-1),
                )
                loss.backward()
                optimizer.step()
            if epoch == 0:
                training_loss, test_loss = self.compute_full_loss(run, device, batch_size)
                print(f"Epoch: {epoch}, Training Loss: {training_loss}, Test Loss: {test_loss}")
                training_loss_vals.append(training_loss)
                test_loss_vals.append(test_loss)
            if (epoch + 1) % 500 == 0:
                training_loss, test_loss = self.compute_full_loss(run, device, batch_size)
                print(f"Epoch: {epoch+1}, Training Loss: {training_loss}, Test Loss: {test_loss}")
                training_loss_vals.append(training_loss)
                test_loss_vals.append(test_loss)

        print(f"Final Training Loss: {training_loss}, Final Test Loss: {test_loss}")
        return training_loss_vals, test_loss_vals


    def compute_full_loss(self, run, device, batch_size):
        criterion = nn.CrossEntropyLoss(reduction="sum")

        training_loss = 0
        for sequence_batch in torch_data.DataLoader(
            dataset=self.training_dataset,
            batch_size=batch_size,
            shuffle=False,
        ):
            sequence_batch = sequence_batch.to(device)
            output = run.model_obj(sequence_batch[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, self.vocab_size),
                sequence_batch[:, 1:].contiguous().view(-1),
            )
            training_loss += loss
        training_loss = training_loss.item()

        test_loss = 0
        for sequence_batch in torch_data.DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
        ):
            sequence_batch = sequence_batch.to(device)
            output = run.model_obj(sequence_batch[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, self.vocab_size),
                sequence_batch[:, 1:].contiguous().view(-1),
            )
            test_loss += loss
        test_loss = test_loss.item()

        return training_loss, test_loss
