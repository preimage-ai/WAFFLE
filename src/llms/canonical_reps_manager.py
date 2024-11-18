from collections import defaultdict
import os
import pickle


class CanonicalRepsManager:
    def __init__(self, run_number, column_name):
        self.run_number = run_number
        self.column_name = column_name
        self.pickle_path = (
            f"data/canonical_reps/{self.column_name}_{self.run_number}.pickle"
        )
        self.canonical_reps = {}

    def get_canonical_rep(self, ans):
        keys_with_ans = [
            key for key, value in self.canonical_reps.items() if ans in value
        ]
        if keys_with_ans:
            return keys_with_ans[0]
        return ans

    def get_canonical_rep_and_update(self, main_ans, secondary_ans):
        # heuristics to avoid adding the same answer twice or to add irrelevant answers
        if main_ans == secondary_ans:
            return main_ans
        if main_ans == "":
            return secondary_ans
        if secondary_ans == "":
            return main_ans

        keys_with_main = [
            key for key, value in self.canonical_reps.items() if main_ans in value
        ]
        keys_with_sec = [
            key for key, value in self.canonical_reps.items() if secondary_ans in value
        ]

        canonical_key = main_ans
        secondary_rep = secondary_ans

        # Check if one of the answers exists in one of the dictionary keys
        if secondary_ans in self.canonical_reps.keys():
            canonical_key = secondary_ans
            secondary_rep = main_ans
        # Check if one of the answers exists in one of the dictionary values
        elif keys_with_main:
            canonical_key = keys_with_main[0]
        elif keys_with_sec:
            canonical_key = keys_with_sec[0]
            secondary_rep = main_ans

        # None of the answers are in the dictionary, add a new entry
        if not keys_with_main and not keys_with_sec:
            self.canonical_reps[canonical_key] = set([secondary_rep])
        else:
            self.canonical_reps[canonical_key].add(secondary_rep)

        return canonical_key

    def save(self):
        # Save dictionary to file
        print("Saving canonical reps to file...")
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.canonical_reps, f)


def combine_pickle_files(pickle_paths, output_pickle_path):
    dicts = []
    for pickle_path in pickle_paths:
        with open(pickle_path, "rb") as f:
            dicts.append(pickle.load(f))

    combined_dict = defaultdict(set)
    for d in dicts:
        for k, v in d.items():
            combined_dict[k] |= v

    result = dict(combined_dict)
    with open(output_pickle_path, "wb") as f:
        pickle.dump(result, f)
