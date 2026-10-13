from _lib.entry import choose

if __name__ == '__main__':
    choose({'single-dyad': 'run_downstream_reciprocity_valence',
            'hiring': 'run_resume_benefactor_hiring',
            'committee': 'run_downstream_benefactor_committee'})
