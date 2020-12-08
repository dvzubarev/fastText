((nil . (
         (eval . (progn
                   (setq-local projectile-project-test-cmd #'helm-ctest)
                   (setq-local projectile-project-compilation-dir "build")
                   (setq-local helm-make-build-dir (projectile-compilation-dir))
                   (setq-local helm-ctest-dir (projectile-compilation-dir))
                   (setq-local cmake-meta-build-dir (projectile-compilation-dir))
                   (setq-local ccls-initialization-options  `(:cache (:directory ,(concat (getenv "HOME") "/.cache/ccls"))
                                                                     :compilationDatabaseDirectory ,(projectile-compilation-dir)))
                   ))
         (projectile-project-name . "FastText")
         (c-basic-offset . 2)
	       (projectile-project-configure-cmd .
					   "cmake -G 'CodeBlocks - Unix Makefiles' \
					   -DCMAKE_BUILD_TYPE=Debug \
					   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..")
	 )))
